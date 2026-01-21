import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import hilbert, welch, find_peaks
import pywt


def compute_ec(v_s, v_w, a_p, w=214):  # w = workpiece OD (mm)
    """Compute adjusted specific energy with CBN-GCr15 empirical model."""
    # Empirical tangential force (Marinescu model for bearing steel)
    F_t = 1.2 * (a_p**0.8) * (v_s**-0.2)  # F_t in Newtons
    e_c = (F_t * v_s) / (a_p * v_w * w)

    # Nonlinear adjustment for v_w (calibrated to your 20-80 rpm range)
    v_w_ref = interp1d([20, 80], [0.85, 1.15])(v_w)  # Scaling factor
    return e_c * v_w_ref


def compute_bdi(a_p):
    """GCr15-specific transition threshold (30µm critical depth)."""
    return a_p / 30.0  # Returns ratio relative to critical depth


def compute_st(v_s, a_p, grinding_time):
    """Max temperature ratio (T_max/T_material) for GCr15."""
    K = 46e-3  # W/mmK
    rho_cp = 3.8e-3  # J/mm³K
    q = 0.8 * v_s * compute_ec(v_s, 40, a_p)  # Heat flux approximation

    def _temp_ode(t, T):
        # Ensure the output is a 1D array
        return np.array([(q * np.sqrt(a_p)) / (K * np.sqrt(np.pi * (K / rho_cp) * t))])

    sol = solve_ivp(_temp_ode, [1e-6, grinding_time], [25], method="Radau")
    # Extract the last temperature value safely
    T_max = sol.y[0, -1] if sol.y.size > 0 else 25
    return T_max / 800.0  # 800°C = GCr15 tempering threshold


def process_vibration(vib_signal, v_s, fs=51.2e3):
    """Analyze wheel-workpiece dynamics for thin-walled bushing."""
    # Envelope analysis (2-5kHz band)
    analytic = hilbert(vib_signal)
    envelope = np.abs(analytic)
    env_kurtosis = np.mean((envelope - np.mean(envelope)) ** 4) / (
        np.std(envelope) ** 4
    )

    # Meshing frequency amplitude (CBN grit count = 400mm⁻1 typical)
    grit_density = 400  # grits/mm
    N_grits = grit_density * np.pi * 400  # Wheel circumference
    f_mesh = (v_s * N_grits) / (60 * 1000)  # Convert v_s mm/s → m/s
    f, Pxx = welch(vib_signal, fs, nperseg=1024)

    # Find frequencies within the range and handle empty arrays
    freq_mask = (f > f_mesh * 0.9) & (f < f_mesh * 1.1)
    if np.any(freq_mask):
        mesh_amp = np.max(Pxx[freq_mask])
    else:
        mesh_amp = 0  # Default value if no frequencies match

    return {"env_kurtosis": env_kurtosis, "mesh_amp": mesh_amp}


def process_ae(ae_signal, fs=4e6, mask_energy=(150e3, 250e3), burst_threshold=4, time_spacing=1e-4):
    """Extract narrowband (150-250kHz) features from 4MHz AE data."""
    # Wavelet energy in fracture-sensitive band
    coeffs, freqs = pywt.cwt(
        ae_signal, np.arange(1, 100), "cmor3-3", sampling_period=1 / fs
    )
    mask = (freqs > mask_energy[0]) & (freqs < mask_energy[1])
    wavelet_energy = np.sum(coeffs[mask] ** 2)

    # Burst detection (threshold = 4σ)
    threshold = burst_threshold * np.std(ae_signal)
    peaks, _ = find_peaks(
        ae_signal, height=threshold, distance=int(fs * time_spacing)
    )  # time_spacing = 0.1ms min spacing
    burst_rate = len(peaks) / (len(ae_signal) / fs)

    return {"wavelet_energy": wavelet_energy, "burst_rate": burst_rate}

def process_triaxial_vib(vib_x, vib_y, vib_z, v_s, fs=51.2e3):
    features = {}
    # Axis-specific envelope kurtosis
    for axis, sig in zip(['x', 'y', 'z'], [vib_x, vib_y, vib_z]):
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        features[f'{axis}_env_kurtosis'] = np.mean((envelope - envelope.mean())**4) / (envelope.std()**4)
    
    # Dominant axis via vector magnitude
    vec_mag = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)
    f, Pxx = welch(vec_mag, fs, nperseg=1024)
    features['mag_mesh_amp'] = np.max(Pxx[(f > 400) & (f < 800)])  # CBN meshing band
    
    return features