1. Physics-Driven Framework
Key Physics for CBN-GCr15 Interaction
Brittle-Ductile Transition: GCr15 steel under CBN grinding exhibits mixed-mode material removal. Critical grinding depth (a_{p,\text{crit}} \approx 30\ \mu m) (empirical for bearing steels) separates brittle fracture (low (a_p)) from plastic flow (high (a_p)).

Specific Grinding Energy:
[
e_c = \frac{F_t \cdot v_s}{a_p \cdot v_w \cdot w} \quad \text{[J/mm³]}
]

(F_t) (tangential force) can be approximated via CBN-specific empirical model:
[
F_t = k \cdot a_p^{0.8} \cdot v_s^{-0.2} \quad \text{((k = 1.2\ \text{N}/\mu m^{0.8}) for GCr15)}
]
Derived from [Marinescu et al., Handbook of Machining with Grinding Wheels, 2016]
Thermal Damage Risk:
[
T_{\text{max}} = \frac{q \cdot a_p^{0.5}}{K \cdot \sqrt{\pi \cdot \kappa}}, \quad \text{where } \kappa = \frac{K}{\rho C_p} \text{ (thermal diffusivity)}
]

For GCr15: (K = 46\ \text{W/mK}), (\rho C_p = 3.8\ \text{J/cm³K}).
2. Feature Engineering Pipeline
A. Physics-Based Features
Brittle-Ductile Indicator:
[
\text{BDI} = \frac{a_p}{a_{p,\text{crit}}} \quad \text{(>1 implies ductile-dominated)}
]
Adjusted Specific Energy:
[
e_c' = e_c \cdot \left(1 + 0.1 \cdot \frac{v_w}{40}\right) \quad \text{(accounts for workpiece speed nonlinearity)}
]
Thermal Severity:
[
S_t = \frac{T_{\text{max}}}{T_{\text{material}}} \quad \text{((T_{\text{material}} = 800^\circ)C for GCr15 tempering threshold)}
]
B. Sensor Signal Features
AE (4 MHz sampling):

Narrowband (50–300 kHz):
Fracture events (spalling in GCr15) → Burst AE energy.
Extract: Wavelet energy in 150–250 kHz band, burst count/sec.
Broadband (1–3 MHz):
Grain wear/microfracture → Continuous AE RMS.
Vibration (51.2 kHz sampling):

Spectral Peaks at (f_{\text{mesh}} = \frac{v_s \cdot N_{\text{grits}}}{\pi \cdot D_w}) (≈400–800 Hz for CBN wheel).
Envelope Analysis of 2–5 kHz band to detect wheel imbalance (critical for thin-walled bushing stability).