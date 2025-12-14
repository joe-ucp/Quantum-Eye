import time
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

try:
    from quantum_eye.adapters.noise_models import get_noise_model_by_type
    _NOISE_IMPORT_STATUS = "ok"
    logger.info("adapters.noise_models import succeeded")
except Exception as exc:  # pragma: no cover - adapter optional
    _NOISE_IMPORT_STATUS = f"failed: {exc}"
    get_noise_model_by_type = None  # type: ignore
    logger.warning("adapters.noise_models import failed: %s", exc)


@dataclass
class VQEResult:
    energy: float
    reference_energy: float
    energy_error: float
    wall_clock_s: float
    shots: int
    iterations: int
    fidelity_to_reference: float
    max_rss_mb: float | None
    success: bool
    extra: Dict[str, Any]


@dataclass
class LiHModel:
    op: SparsePauliOp
    matrix: np.ndarray
    ground_energy: float
    ground_state: np.ndarray
    pauli_mats: List[np.ndarray]
    diag_masks: List[np.ndarray]
    diag_terms: List[bool]
    labels: List[str]
    coeffs: List[float]


def _lih_terms() -> List[Tuple[str, float]]:
    """
    Fixed 6-qubit LiH-inspired Hamiltonian (parity-mapped and padded to 6 qubits).
    Coefficients are deterministic and chosen to include off-diagonal terms so that
    single-basis mitigation has an effect.
    """
    return [
        ("IIIIII", -7.498000),
        ("ZIIIII", 0.196000),
        ("IZIIII", 0.196000),
        ("IIZIII", 0.098000),
        ("IIIZII", 0.045000),
        ("IIIIZI", -0.030000),
        ("IIIIIZ", 0.020000),
        ("ZZIIII", -0.120000),
        ("IZZIII", 0.080000),
        ("IIZZII", 0.050000),
        ("IIIZZI", -0.070000),
        ("IIIIZZ", 0.040000),
        ("XXIIII", 0.150000),
        ("YYIIII", 0.150000),
        ("IIXXII", 0.120000),
        ("IIYYII", 0.120000),
        ("IIIIXX", 0.080000),
        ("IIIIYY", 0.080000),
        ("IXXIII", 0.050000),
        ("IYYIII", 0.050000),
        ("IIXXZI", -0.040000),
        ("IIYYZI", -0.040000),
    ]


_MODEL: LiHModel | None = None


def _build_model() -> LiHModel:
    terms = _lih_terms()
    labels = [lbl for (lbl, _) in terms]
    coeffs = [coeff for (_, coeff) in terms]
    op = SparsePauliOp.from_list(list(zip(labels, coeffs)))
    matrix = op.to_matrix()
    eigvals, eigvecs = np.linalg.eigh(matrix)
    ground_idx = int(np.argmin(eigvals))
    ground_energy = float(np.real(eigvals[ground_idx]))
    ground_state = np.array(eigvecs[:, ground_idx]).flatten()

    pauli_mats: List[np.ndarray] = []
    diag_masks: List[np.ndarray] = []
    diag_terms: List[bool] = []
    for label in labels:
        pauli = SparsePauliOp(label, coeffs=[1.0]).to_matrix()
        pauli_mats.append(pauli)
        is_diag = "X" not in label and "Y" not in label
        diag_terms.append(is_diag)
        if is_diag:
            mask = _parity_mask(label, num_qubits=6)
        else:
            mask = np.array([])
        diag_masks.append(mask)

    return LiHModel(
        op=op,
        matrix=matrix,
        ground_energy=ground_energy,
        ground_state=ground_state,
        pauli_mats=pauli_mats,
        diag_masks=diag_masks,
        diag_terms=diag_terms,
        labels=labels,
        coeffs=coeffs,
    )


def _get_model() -> LiHModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


def _parity_mask(label: str, num_qubits: int) -> np.ndarray:
    size = 2**num_qubits
    mask = np.ones(size, dtype=float)
    for bit in range(num_qubits):
        if label[-1 - bit] == "Z":
            # Flip sign when bit is 1 at this position
            indices = np.arange(size)
            ones = ((indices >> bit) & 1).astype(float)
            mask *= np.where(ones > 0, -1.0, 1.0)
    return mask


def _counts_to_prob(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
    total = max(1, sum(counts.values()))
    probs = np.zeros(2**num_qubits, dtype=float)
    for bitstring, ct in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs[idx] = ct / total
    return probs


def _compute_features(probs: np.ndarray) -> Dict[str, float]:
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    variance = float(np.var(probs))
    ipr = float(np.sum(probs**2))
    state_distribution = float(np.exp(-abs(ipr - 2.0)))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    even = float(probs[::2].sum())
    odd = float(probs[1::2].sum())
    correlations = float(abs(even - odd))
    return {
        "P": variance,
        "S": state_distribution,
        "E": entropy,
        "Q": correlations,
    }


def _qwc_compatible(label: str, group: List[str]) -> bool:
    """Return True if label qubit-wise commutes with all labels in group."""
    for other in group:
        for a, b in zip(label, other):
            if a == "I" or b == "I":
                continue
            if a != b:
                return False
    return True


def _group_paulis_qwc(labels: List[str], coeffs: List[float]) -> List[List[Tuple[str, float]]]:
    """Greedy qubit-wise commuting grouping (simple and stable)."""
    groups: List[List[Tuple[str, float]]] = []
    for label, coeff in zip(labels, coeffs):
        placed = False
        for grp in groups:
            if _qwc_compatible(label, [g[0] for g in grp]):
                grp.append((label, coeff))
                placed = True
                break
        if not placed:
            groups.append([(label, coeff)])
    return groups


def _apply_basis_rotations(qc: QuantumCircuit, label: str) -> None:
    """Rotate measurement basis so group Pauli terms are diagonal in Z."""
    num_qubits = qc.num_qubits
    for i, pauli in enumerate(label):
        if pauli == "X":
            qc.h(num_qubits - 1 - i)
        elif pauli == "Y":
            qc.sdg(num_qubits - 1 - i)
            qc.h(num_qubits - 1 - i)
        # Z or I: no rotation


def _expectation_from_counts(label: str, counts: Dict[str, int], num_qubits: int) -> float:
    """Compute <label> from bitstring counts measured in Z basis."""
    total = max(1, sum(counts.values()))
    exp_val = 0.0
    for bitstring, ct in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 0
        for idx, pauli in enumerate(reversed(label)):  # rightmost bit is qubit 0
            if pauli == "I":
                continue
            if bits[-1 - idx] == "1":
                parity ^= 1
        contrib = -1.0 if parity else 1.0
        exp_val += contrib * ct
    return exp_val / total


def _expectation_full_baseline(
    theta: np.ndarray,
    backend: AerSimulator,
    model: LiHModel,
    shots_total_target: int,
    rng: np.random.Generator,
) -> Tuple[float, Dict[str, float]]:
    """
    Full-Hamiltonian expectation via grouped Pauli measurements (QWC) under noise.
    Shots are distributed equally across groups to keep budgets comparable.
    """
    groups = _group_paulis_qwc(model.labels, model.coeffs)
    num_groups = max(1, len(groups))
    base = max(1, shots_total_target // num_groups)
    remainder = shots_total_target - base * num_groups
    energy = 0.0
    total_used = 0

    for gi, group in enumerate(groups):
        group_shots = base + (1 if gi < remainder else 0)
        total_used += group_shots

        qc = _ansatz(theta)
        # rotate basis common to all terms in the group
        _apply_basis_rotations(qc, group[0][0])
        qc.measure_all()

        local_seed = int(rng.integers(0, 1_000_000_000))
        compiled = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=local_seed)
        result = backend.run(compiled, shots=group_shots, seed_simulator=local_seed).result()
        counts = result.get_counts()
        if isinstance(counts, list):  # pragma: no cover - legacy qiskit path
            counts_dict: Dict[str, int] = {}
            for entry in counts:
                counts_dict.update(entry)
            counts = counts_dict

        for label, coeff in group:
            exp_val = _expectation_from_counts(label, counts, num_qubits=6)
            energy += coeff * exp_val

    feats = {
        "estimator": "full_baseline",
        "groups": num_groups,
        "shots_total_target": shots_total_target,
        "shots_used": total_used,
    }
    return energy, feats


def _quantum_eye_enhance(
    probs: np.ndarray, smoothing_window: float, uniform_floor: float
) -> np.ndarray:
    """Frequency smoothing to denoise single-basis counts (stronger for low-shot)."""
    probs = np.clip(probs, 0.0, None)
    if probs.sum() == 0:
        return np.ones_like(probs) / probs.size
    probs = probs / probs.sum()
    freq = np.fft.fft(probs)
    freqs = np.fft.fftfreq(probs.size)
    window = np.exp(-smoothing_window * (freqs**2))
    smoothed = np.real(np.fft.ifft(freq * window))
    smoothed = np.clip(smoothed, 0.0, None)
    # Mix a small uniform floor to stabilize SPSA gradients under low shots.
    smoothed = (1.0 - uniform_floor) * smoothed + uniform_floor * (
        np.ones_like(smoothed) / smoothed.size
    )
    if smoothed.sum() <= 0:
        return probs
    return smoothed / smoothed.sum()


def _expectation_baseline(probs: np.ndarray, model: LiHModel) -> float:
    energy = 0.0
    for coeff, is_diag, mask in zip(model.coeffs, model.diag_terms, model.diag_masks):
        if not is_diag:
            continue
        term_val = float(np.dot(probs, mask))
        energy += coeff * term_val
    return energy


def _expectation_qe(
    probs: np.ndarray, model: LiHModel, smoothing_window: float, uniform_floor: float
) -> Tuple[float, Dict[str, float]]:
    enhanced = _quantum_eye_enhance(
        probs, smoothing_window=smoothing_window, uniform_floor=uniform_floor
    )
    amps = np.sqrt(enhanced + 1e-12)
    energy = 0.0
    for coeff, mat in zip(model.coeffs, model.pauli_mats):
        val = float(np.real(np.conj(amps) @ (mat @ amps)))
        energy += coeff * val
    features = _compute_features(enhanced)
    return energy, features


def _expectation_cached(
    probs: np.ndarray,
    model: LiHModel,
    cache: "ConstraintCache",
    cache_threshold: float = 0.1,
    smoothing_window: float = 45.0,
    uniform_floor: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """
    Cache-aware expectation computation. Uses cached bounds to skip/reuse measurements.
    
    Args:
        probs: Probability distribution from Z-basis measurements
        model: LiH model
        cache: ConstraintCache with measurement bounds
        cache_threshold: CI width threshold below which to use cached estimate
        smoothing_window: QE smoothing parameter (if using QE for remaining terms)
        uniform_floor: QE uniform floor parameter
    
    Returns:
        (energy, features_dict) where features includes shot usage stats
    """
    from constraint_cache import ConstraintCache
    
    energy = 0.0
    terms_skipped = 0
    terms_cached = 0
    terms_measured = 0
    shots_saved = 0
    
    # Enhanced probabilities for off-diagonal terms (if needed)
    enhanced = _quantum_eye_enhance(
        probs, smoothing_window=smoothing_window, uniform_floor=uniform_floor
    )
    amps = np.sqrt(enhanced + 1e-12)
    
    for label, coeff, is_diag, mask, mat in zip(
        model.labels, model.coeffs, model.diag_terms, model.diag_masks, model.pauli_mats
    ):
        # Check if term is forbidden (provably zero)
        if label in cache.forbidden_terms:
            terms_skipped += 1
            # Energy contribution is 0, skip
            continue
        
        # Check if term has cached bounds
        if label in cache.measurement_bounds:
            lo, hi = cache.measurement_bounds[label]
            ci_width = hi - lo
            
            # If CI is tight enough, use cached estimate (midpoint)
            if ci_width < cache_threshold:
                cached_val = (lo + hi) / 2.0
                energy += coeff * cached_val
                terms_cached += 1
                # Estimate shots saved (would need to measure this term)
                shots_saved += 1  # Simplified: 1 term = 1 measurement
                continue
        
        # Otherwise, measure term normally
        terms_measured += 1
        if is_diag:
            # Diagonal term: use probs directly
            term_val = float(np.dot(probs, mask))
        else:
            # Off-diagonal term: use enhanced state
            term_val = float(np.real(np.conj(amps) @ (mat @ amps)))
        energy += coeff * term_val
    
    features = {
        "estimator": "cached",
        "terms_skipped": terms_skipped,
        "terms_cached": terms_cached,
        "terms_measured": terms_measured,
        "shots_saved_estimate": shots_saved,
        "cache_threshold": cache_threshold,
    }
    
    return energy, features


def _reconstruct_psqe_state(
    probs: np.ndarray, smoothing_window: float, uniform_floor: float
) -> np.ndarray:
    """
    Heuristic PSQE-style reconstruction: apply frequency-domain smoothing and keep
    complex phases from the spectrum to recover off-diagonal sensitivity.
    """
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        size = probs.size if probs.size > 0 else 1
        return np.ones(size, dtype=complex) / np.sqrt(size)
    probs = probs / probs.sum()
    # soften zeros
    probs = (1.0 - uniform_floor) * probs + uniform_floor * (
        np.ones_like(probs) / probs.size
    )
    freqs = np.fft.fftfreq(probs.size)
    window = np.exp(-smoothing_window * (freqs**2))
    spectrum = np.fft.fft(probs)
    smoothed = spectrum * window
    amps = np.fft.ifft(smoothed)
    norm = np.linalg.norm(amps)
    if norm <= 1e-12:
        return np.sqrt(probs + 1e-12)
    return amps / norm


def _psqe_coherence(probs: np.ndarray) -> float:
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        return 0.0
    probs = probs / probs.sum()
    spectrum = np.fft.fft(probs)
    flat = spectrum.flatten()
    mag_var = float(np.var(np.abs(flat)))
    phase_var = float(np.var(np.angle(flat)))
    peak = float(np.max(np.abs(flat)))
    avg = float(np.mean(np.abs(flat)) + 1e-10)
    par = peak / avg
    coherence = np.exp(-mag_var) * np.exp(-phase_var) * (par / 10.0)
    return float(np.clip(coherence, 0.0, 1.0))


def _expectation_psqe_fft(
    probs: np.ndarray, model: LiHModel, smoothing_window: float, uniform_floor: float
) -> Tuple[float, Dict[str, float]]:
    amps = _reconstruct_psqe_state(
        probs, smoothing_window=smoothing_window, uniform_floor=uniform_floor
    )
    energy = 0.0
    for coeff, mat in zip(model.coeffs, model.pauli_mats):
        val = float(np.real(np.conj(amps) @ (mat @ amps)))
        energy += coeff * val
    feats = _compute_features(np.abs(amps) ** 2)
    feats["psqe_coherence"] = _psqe_coherence(np.abs(amps) ** 2)
    return energy, feats


def _expectation_psqe(probs: np.ndarray, model: LiHModel, sweeps: int = 3) -> Tuple[float, Dict[str, float]]:
    """
    PSQE-inspired post-processing: infer sign pattern that minimizes energy given
    measured probabilities. This is a classical Ising-like solve over +/- phases.
    """
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0:
        return float("nan"), {"psqe_flips": 0, "psqe_sweeps": sweeps}
    probs = probs / probs.sum()
    sqrtp = np.sqrt(probs + 1e-12)
    H = np.real(model.matrix)
    # A_ij = sqrt(p_i) * H_ij * sqrt(p_j); symmetric
    A = sqrtp[:, None] * H * sqrtp[None, :]
    n = A.shape[0]
    s = np.ones(n, dtype=float)

    def energy(signs: np.ndarray) -> float:
        return float(signs @ (A @ signs))

    flips = 0
    for _ in range(sweeps):
        for i in range(n):
            # energy delta for flipping s_i in Ising quadratic form
            h_i = A[i, :] @ s
            delta = -4.0 * s[i] * h_i
            if delta < 0.0:
                s[i] *= -1.0
                flips += 1
    e_val = energy(s)
    return e_val, {"psqe_flips": flips, "psqe_sweeps": sweeps}


def _ansatz(theta: np.ndarray, layers: int = 2) -> QuantumCircuit:
    num_qubits = 6
    qc = QuantumCircuit(num_qubits)
    # Hartree-Fock-style init: occupy first three qubits
    for q in range(3):
        qc.x(q)
    param_idx = 0
    for _layer in range(layers):
        for q in range(num_qubits):
            qc.ry(theta[param_idx], q)
            param_idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    return qc


def _sample_counts(
    theta: np.ndarray, shots: int, seed: int, backend: AerSimulator
) -> Dict[str, int]:
    qc = _ansatz(theta)
    qc.measure_all()
    compiled = transpile(
        qc, backend=backend, optimization_level=1, seed_transpiler=seed
    )
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    if isinstance(counts, list):  # pragma: no cover - legacy qiskit path
        counts_dict: Dict[str, int] = {}
        for entry in counts:
            counts_dict.update(entry)
        return counts_dict
    return counts  # type: ignore[return-value]


def _fidelity_to_ground(theta: np.ndarray, model: LiHModel) -> float:
    state = Statevector.from_instruction(_ansatz(theta))
    overlap = np.vdot(model.ground_state, state.data)
    return float(np.abs(overlap) ** 2)


def _current_memory_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def _spsa_schedule(a0: float, c0: float, k: int) -> Tuple[float, float]:
    ak = a0 / ((k + 1) ** 0.602)
    ck = c0 / ((k + 1) ** 0.101)
    return ak, ck


def build_lih_cache(
    seed: int = 12345,
    shots: int = 8192,
    protocol: str = "z",
    source: str = "exact",
    vqe_theta: np.ndarray | None = None,
) -> "ConstraintCache":
    """
    Build constraint cache from LiH ground state measurements.
    
    Args:
        seed: Random seed for measurement sampling
        shots: Number of shots for measurements
        protocol: "z" for Z-basis only, "z+x" for Z and X rotations
        source: "exact" for exact ground state, "vqe" for VQE-produced state
        vqe_theta: Parameter vector for VQE state (required if source="vqe")
    
    Returns:
        ConstraintCache with term_support, measurement_bounds, symmetry_sectors
    """
    from constraint_cache import ConstraintCache
    from hamiltonian_elimination import (
        expectation_from_state,
        measurement_bounds_from_state,
        measurement_bounds_from_counts,
        sample_z_counts,
        parity_from_state,
        parity_from_counts,
    )
    
    rng = np.random.default_rng(seed)
    model = _get_model()
    num_qubits = 6
    
    # Get ground state
    if source == "exact":
        ground_state = model.ground_state
    elif source == "vqe":
        if vqe_theta is None:
            raise ValueError("vqe_theta required when source='vqe'")
        state = Statevector.from_instruction(_ansatz(vqe_theta))
        ground_state = np.array(state.data)
    else:
        raise ValueError(f"Unknown source: {source}")
    
    # Get all LiH terms
    lih_terms = _lih_terms()
    term_labels = [label for (label, _) in lih_terms]
    term_support = set(term_labels)
    
    # Sample Z-basis measurements
    z_counts = sample_z_counts(ground_state, shots=shots, n_qubits=num_qubits, rng=rng)
    
    # Compute measurement bounds for diagonal terms (Z-basis)
    diag_observables = [label for label in term_labels if "X" not in label and "Y" not in label]
    measurement_bounds, measured_expectations = measurement_bounds_from_counts(
        z_counts, diag_observables, shots
    )
    
    # If protocol includes X rotations, add bounds for X/Y terms
    if protocol == "z+x":
        # Sample X-basis measurements (H on all qubits, then measure Z)
        qc_x = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            qc_x.h(q)
        qc_x.measure_all()
        
        # Prepare state and sample
        state_x = Statevector(ground_state)
        state_x = state_x.evolve(qc_x)
        probs_x = np.abs(state_x.data) ** 2
        probs_x = probs_x / probs_x.sum()
        outcomes_x = rng.choice(len(probs_x), size=shots, p=probs_x)
        x_counts: Dict[str, int] = {}
        for idx in outcomes_x:
            bitstring = format(idx, f"0{num_qubits}b")
            x_counts[bitstring] = x_counts.get(bitstring, 0) + 1
        
        # Get X/Y observables
        offdiag_observables = [label for label in term_labels if "X" in label or "Y" in label]
        # For X-basis measurements, we can compute expectations for X terms
        x_observables = [label for label in offdiag_observables if "Y" not in label]
        if x_observables:
            x_bounds, x_expectations = measurement_bounds_from_counts(
                x_counts, x_observables, shots
            )
            measurement_bounds.update(x_bounds)
            measured_expectations.update(x_expectations)
    
    # Compute exact expectations for all terms (for reference)
    exact_expectations = {
        label: expectation_from_state(label, ground_state) for label in term_labels
    }
    
    # Determine parity symmetry
    measured_parity = parity_from_counts(z_counts, shots)
    exact_parity = parity_from_state(ground_state, num_qubits)
    parity_sector = measured_parity if measured_parity in ("even", "odd") else exact_parity
    
    # Identify forbidden terms (provably zero by symmetry or structure)
    forbidden_terms: set[str] = set()
    # Terms that are identically zero can be added here if known
    
    # Build cache
    cache = ConstraintCache(
        term_support=term_support,
        forbidden_terms=forbidden_terms,
        symmetry_sectors={"parity": parity_sector} if parity_sector in ("even", "odd") else {},
        reachability={},  # Not used for LiH cache
        measurement_bounds=measurement_bounds,
        metadata={
            "n_qubits": num_qubits,
            "shots": shots,
            "seed": seed,
            "protocol": protocol,
            "source": source,
            "lih_instance_id": hash(tuple(sorted(term_labels))),
            "num_terms": len(term_labels),
            "num_diag_terms": len(diag_observables),
            "measured_parity": measured_parity,
            "exact_parity": exact_parity,
        },
    )
    
    return cache


def _prepare_noise_model(
    noise_type: str | None, noise_level: float, num_qubits: int
) -> Tuple[Any, Dict[str, Any]]:
    """Build noise model if available and log details."""
    info: Dict[str, Any] = {
        "noise_type": noise_type,
        "noise_level": noise_level,
        "import_status": _NOISE_IMPORT_STATUS,
    }
    if noise_level <= 0 or not noise_type:
        info["status"] = "disabled"
        logger.info(
            "[noise] disabled (type=%s, level=%.3f)", noise_type, noise_level
        )
        return None, info

    if get_noise_model_by_type is None:
        info["status"] = "import-missing"
        logger.warning(
            "[noise] requested but adapters.noise_models is unavailable "
            "(type=%s, level=%.3f)",
            noise_type,
            noise_level,
        )
        return None, info

    try:
        noise_model = get_noise_model_by_type(
            noise_type=noise_type,
            noise_level=noise_level,
            metadata={"num_qubits": num_qubits},
        )
        instructions = len(getattr(noise_model, "noise_instructions", []))
        basis = getattr(noise_model, "basis_gates", None)
        info["noise_instructions"] = instructions
        info["basis_gates"] = basis
        info["status"] = "ok" if instructions else "empty"
        logger.info(
            "[noise] built model type=%s level=%.3f instructions=%s basis=%s",
            noise_type,
            noise_level,
            instructions,
            basis,
        )
        if instructions == 0:
            logger.warning(
                "[noise] noise model has no noise_instructions; simulator will be noiseless"
            )
        return noise_model, info
    except Exception as exc:  # pragma: no cover - diagnostic path
        info["status"] = f"build-error:{exc}"
        logger.warning(
            "[noise] failed to build model type=%s level=%.3f error=%s",
            noise_type,
            noise_level,
            exc,
        )
        return None, info


def run_vqe_lih(
    mode: str,
    seed: int = 12345,
    shots: int = 2048,
    max_iter: int = 40,
    layers: int = 2,
    spsa_a0: float | None = None,
    spsa_c0: float | None = None,
    qe_smoothing_window: float = 45.0,
    qe_uniform_floor: float = 0.1,
    estimator: str = "qe",
    noise_type: str | None = None,
    noise_level: float = 0.0,
    cache_path: Path | None = None,
) -> VQEResult:
    """
    Shot-based VQE on a fixed 6-qubit LiH Hamiltonian.

    Baseline: estimates energy from Z-basis counts using only diagonal terms.
    Adapter: applies Quantum Eye frequency smoothing + state estimation to include
    off-diagonal contributions from single-basis counts.
    
    If cache_path is provided, uses cache-aware estimator to skip/reuse measurements.
    """
    from constraint_cache import ConstraintCache
    
    model = _get_model()
    rng = np.random.default_rng(seed)
    num_params = layers * 6
    theta = rng.normal(scale=0.2, size=num_params)
    
    # Load cache if provided (must be defined before nested function to be captured)
    _cache: ConstraintCache | None = None
    if cache_path is not None:
        # Normalize path: handle both string and Path, and resolve forward slashes on Windows
        cache_path_normalized = Path(cache_path)
        if not cache_path_normalized.is_absolute():
            # If relative, try to resolve it
            cache_path_normalized = cache_path_normalized.resolve()
        logger.info(f"[cache] attempting to load cache from: {cache_path_normalized}")
        logger.info(f"[cache] path exists: {cache_path_normalized.exists()}")
        if not cache_path_normalized.exists():
            logger.warning(f"[cache] cache file not found: {cache_path_normalized}, proceeding without cache")
        else:
            try:
                _cache = ConstraintCache.from_json(cache_path_normalized)
                logger.info(f"[cache] loaded cache from {cache_path_normalized}")
                logger.info(f"[cache] term_support: {len(_cache.term_support)} terms")
                logger.info(f"[cache] measurement_bounds: {len(_cache.measurement_bounds)} bounds")
            except Exception as exc:
                logger.warning(f"[cache] failed to load cache: {exc}, proceeding without cache")
    cache = _cache  # Make available to nested function

    noise_model, noise_info = _prepare_noise_model(
        noise_type=noise_type, noise_level=noise_level, num_qubits=6
    )
    if noise_model is not None and getattr(noise_model, "noise_instructions", []):
        backend_sample = AerSimulator(
            noise_model=noise_model,
            basis_gates=getattr(noise_model, "basis_gates", None),
        )
        backend_noise = getattr(backend_sample.options, "noise_model", None)
        logger.info(
            "[noise] AerSimulator configured with noise_model set=%s",
            bool(backend_noise),
        )
    else:
        backend_sample = AerSimulator()
        if noise_info.get("status") not in {"disabled"}:
            logger.warning(
                "[noise] requested but running noiseless (status=%s)",
                noise_info.get("status"),
            )
        else:
            logger.info("[noise] using noiseless AerSimulator")

    total_shots = 0
    energy_trace: List[float] = []
    mitigation_trace: List[Dict[str, float]] = []
    counts_trace: List[Dict[str, int]] = []
    t0 = time.perf_counter()

    def eval_energy(current_theta: np.ndarray) -> Tuple[float, Dict[str, float]]:
        nonlocal total_shots, cache
        if mode == "adapter" and estimator == "full_baseline":
            raise ValueError("full_baseline estimator is not valid in adapter mode")

        if mode == "adapter":
            local_seed = int(rng.integers(0, 1_000_000_000))
            counts = _sample_counts(
                current_theta, shots=shots, seed=local_seed, backend=backend_sample
            )
            total_shots += shots
            probs = _counts_to_prob(counts, num_qubits=6)

            # Check for cache-aware estimators
            if cache is not None and estimator in ("qe_cached", "baseline_cached"):
                logger.info(f"[cache] using cache-aware estimator: {estimator}")
                energy_val, feats = _expectation_cached(
                    probs,
                    model,
                    cache,
                    cache_threshold=0.1,
                    smoothing_window=qe_smoothing_window,
                    uniform_floor=qe_uniform_floor,
                )
                feats["estimator"] = estimator
                feats["counts"] = counts
                return energy_val, feats
            elif estimator in ("qe_cached", "baseline_cached"):
                logger.warning(f"[cache] cache-aware estimator {estimator} requested but cache is None, falling back to baseline")

            if estimator == "psqe":
                energy_val, feats = _expectation_psqe(probs, model)
                feats["estimator"] = "psqe"
                feats["counts"] = counts
                return energy_val, feats
            if estimator == "psqe_fft":
                energy_val, feats = _expectation_psqe_fft(
                    probs,
                    model,
                    smoothing_window=qe_smoothing_window,
                    uniform_floor=qe_uniform_floor,
                )
                feats["estimator"] = "psqe_fft"
                feats["counts"] = counts
                return energy_val, feats
            if estimator == "qe":
                energy_val, feats = _expectation_qe(
                    probs,
                    model,
                    smoothing_window=qe_smoothing_window,
                    uniform_floor=qe_uniform_floor,
                )
                feats["estimator"] = "qe"
                feats["counts"] = counts
                return energy_val, feats
            # Fallback adapter path
            energy_val = _expectation_baseline(probs, model)
            return energy_val, {"estimator": "baseline", "counts": counts}

        # Baseline (including full_baseline)
        if estimator == "full_baseline":
            energy_val, feats = _expectation_full_baseline(
                current_theta,
                backend=backend_sample,
                model=model,
                shots_total_target=shots,
                rng=rng,
            )
            total_shots += int(feats.get("shots_used", 0))
            return energy_val, feats

        local_seed = int(rng.integers(0, 1_000_000_000))
        counts = _sample_counts(
            current_theta, shots=shots, seed=local_seed, backend=backend_sample
        )
        total_shots += shots
        probs = _counts_to_prob(counts, num_qubits=6)
        
        # Check for cache-aware baseline estimator
        if cache is not None and estimator == "baseline_cached":
            energy_val, feats = _expectation_cached(
                probs,
                model,
                cache,
                cache_threshold=0.1,
                smoothing_window=qe_smoothing_window,
                uniform_floor=qe_uniform_floor,
            )
            feats["estimator"] = "baseline_cached"
            feats["counts"] = counts
            return energy_val, feats
        
        energy_val = _expectation_baseline(probs, model)
        return energy_val, {"estimator": "baseline", "counts": counts}

    # SPSA loop
    if mode == "adapter":
        a0_default, c0_default = 0.18, 0.08
    else:
        a0_default, c0_default = 0.22, 0.12

    a0 = a0_default if spsa_a0 is None else spsa_a0
    c0 = c0_default if spsa_c0 is None else spsa_c0
    for k in range(max_iter):
        ak, ck = _spsa_schedule(a0, c0, k)
        delta = rng.choice([-1.0, 1.0], size=num_params)
        e_plus, feats_plus = eval_energy(theta + ck * delta)
        e_minus, feats_minus = eval_energy(theta - ck * delta)
        if "counts" in feats_plus:
            counts_trace.append(feats_plus["counts"])
            feats_plus = {k: v for k, v in feats_plus.items() if k != "counts"}
        if "counts" in feats_minus:
            counts_trace.append(feats_minus["counts"])
            feats_minus = {k: v for k, v in feats_minus.items() if k != "counts"}
        ghat = (e_plus - e_minus) / (2 * ck * delta)
        theta = theta - ak * ghat
        energy_trace.append(e_plus)
        if feats_plus:
            mitigation_trace.append(feats_plus)

    final_energy, final_feats = eval_energy(theta)
    energy_trace.append(final_energy)
    if final_feats:
        if "counts" in final_feats:
            counts_trace.append(final_feats["counts"])
            final_feats = {k: v for k, v in final_feats.items() if k != "counts"}
        mitigation_trace.append(final_feats)

    wall_clock = time.perf_counter() - t0
    fidelity = _fidelity_to_ground(theta, model)
    mem_mb = _current_memory_mb()

    return VQEResult(
        energy=final_energy,
        reference_energy=model.ground_energy,
        energy_error=final_energy - model.ground_energy,
        wall_clock_s=wall_clock,
        shots=total_shots,
        iterations=max_iter,
        fidelity_to_reference=fidelity,
        max_rss_mb=mem_mb,
        success=True,
        extra={
            "energy_trace": energy_trace,
            "mitigation_trace": mitigation_trace,
            "counts_trace": counts_trace,
            "ground_energy": model.ground_energy,
            "mode": mode,
            "seed": seed,
            "shots_per_eval": shots,
            "layers": layers,
            "qe_smoothing_window": qe_smoothing_window,
            "qe_uniform_floor": qe_uniform_floor,
            "spsa_a0": a0,
            "spsa_c0": c0,
            "estimator": estimator,
            "noise_type": noise_type,
            "noise_level": noise_level,
            "noise_debug": noise_info,
        },
    )