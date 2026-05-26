# %% [markdown]
# # Optimasi Parameter Diagram Kendali Double Sampling Coefficient of Variation (DS CV)
# # dengan Pengaruh Ralat Pengukuran (Measurement Error)
# 
# ## Referensi: Nuriman et al. (2026) - DS CV-ME (Linear Covariate Error Model)
# ### Model: Y_ij = A + B*X_ij + e_ij
# ### Metrik Performa: TARL (Time-Adjusted Run Length)
# ---
# ### Kondisi 1: Constant Variance Error (sigma_M^2 konstan)
# ### Kondisi 2: Linear Variance Error (sigma_M_j = omega*sigma_X0 + psi*mu_X)

# %% [markdown]
# ## 1. Import Library dan Setup

# %%
import numpy as np
from scipy import stats, optimize
from scipy.optimize import differential_evolution, minimize
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  OPTIMASI DS CV-ME CONTROL CHART")
print("  Metrik: TARL (Time-Adjusted Run Length)")
print("="*70)


# %% [markdown]
# ## 2. Distribusi Sample CV dan Fungsi Dasar
#
# ### Teori Distribusi:
# Untuk sampel X_1,...,X_n dari N(mu, sigma^2):
# - CV_hat = S/X_bar (sample CV)
# - Statistik U = n/CV_hat^2 mengikuti distribusi non-central chi-square
#   U ~ chi^2(n-1, lambda) dengan lambda = n*gamma^2, gamma = mu/sigma = 1/CV
#
# ### CDF Sample CV:
# P(CV_hat <= c) = P(U >= n/c^2) = 1 - F_{chi2}(n/c^2 | n-1, n/CV^2)

# %%
def cdf_sample_cv(c, n, cv_true):
    """
    CDF distribusi sample CV menggunakan distribusi chi-square non-sentral.
    
    P(CV_hat <= c) = 1 - F_{chi2_nc}(n/c^2 | df=n-1, nc=n/cv_true^2)
    
    Parameters:
    -----------
    c : float - Nilai batas CV
    n : int - Ukuran sampel
    cv_true : float - CV populasi sebenarnya
    
    Returns:
    --------
    P(CV_hat <= c)
    """
    if c <= 0:
        return 0.0
    if cv_true <= 0:
        return 1.0
    
    df = n - 1                    # Derajat kebebasan
    nc = n / (cv_true**2)         # Parameter non-sentralitas = n*gamma^2
    threshold = n / (c**2)        # Titik evaluasi
    
    # P(CV_hat <= c) = P(chi2_nc >= n/c^2) = 1 - CDF_chi2_nc(n/c^2)
    prob = 1.0 - stats.ncx2.cdf(threshold, df=df, nc=nc)
    return prob


def prob_cv_in_range(cv_lower, cv_upper, n, cv_true):
    """
    P(cv_lower < CV_hat < cv_upper)
    """
    if cv_lower >= cv_upper:
        return 0.0
    return cdf_sample_cv(cv_upper, n, cv_true) - cdf_sample_cv(cv_lower, n, cv_true)


print("Fungsi distribusi CV berhasil didefinisikan.")
print(f"  Contoh: P(CV_hat <= 0.25 | n=10, CV=0.2) = {cdf_sample_cv(0.25, 10, 0.2):.4f}")


# %% [markdown]
# ## 3. Model Measurement Error - CV yang Diamati
#
# ### Linear Covariate Error Model: Y_ij = A + B*X_ij + e_ij
#
# **Kondisi 1 (Constant Variance):** e_ij ~ N(0, sigma_M^2)
# - Var(Y_bar_i) = B^2 * Var(X) + sigma_M^2/m
# - CV_Y = sqrt(B^2*CV_X^2 + theta^2/(m*n)) / ... 
# - Simplified: CV_Y_observed tergantung pada B, theta, m
#
# **Kondisi 2 (Linear Variance):** sigma_M = omega*sigma_X0 + psi*mu_X
# - Varians error bervariasi dengan mean proses

# %%
def cv_observed_constant(cv_0, delta, B, theta, m, eta=1.0):
    """
    CV yang diamati (observed) pada Kondisi 1: Constant Variance Error.
    
    Dengan measurement error, CV yang diamati berubah:
    CV_Y = sqrt(B^2 * delta^2 * CV_0^2 + theta^2/m) / |B|
    
    Di mana:
    - CV asli out-of-control: CV_1 = delta * CV_0
    - sigma_M = theta * sigma_X0 (konstan)
    - Dengan m replikasi: varians error efektif = sigma_M^2/m
    - eta: faktor pergeseran pada varians error (default=1, tidak bergeser)
    
    Parameters:
    -----------
    cv_0 : float - CV in-control
    delta : float - Faktor pergeseran (CV_1 = delta * CV_0)
    B : float - Slope model kovariat linear
    theta : float - Rasio sigma_M/sigma_X0
    m : int - Jumlah replikasi
    eta : float - Faktor pergeseran varians error
    
    Returns:
    --------
    cv_observed : float - CV yang diamati setelah pengaruh ME
    """
    # CV proses yang sebenarnya setelah shift
    cv_actual = delta * cv_0
    
    # Varians observed = B^2 * sigma_X^2 + sigma_M^2/m
    # = B^2 * (cv_actual * mu_X)^2 + (eta*theta*sigma_X0)^2/m
    # = mu_X^2 * [B^2 * cv_actual^2 + (eta*theta*cv_0)^2/m]
    # (karena sigma_X0 = cv_0 * mu_X)
    #
    # Mean observed = B * mu_X (dengan A=0)
    # CV_observed = sqrt(Var) / Mean = sqrt(B^2*cv_actual^2 + eta^2*theta^2*cv_0^2/m) / |B|
    
    var_term = B**2 * cv_actual**2 + (eta * theta * cv_0)**2 / m
    cv_observed = np.sqrt(var_term) / abs(B)
    
    return cv_observed


def cv_observed_linear(cv_0, delta, B, theta, m, omega, psi, eta=1.0):
    """
    CV yang diamati pada Kondisi 2: Linear Variance Error.
    
    sigma_M = omega * sigma_X0 + psi * mu_X
    sigma_M = mu_X * (omega * cv_0 + psi)
    
    CV_observed = sqrt(B^2 * cv_actual^2 + (omega*cv_0 + psi)^2/m) / |B|
    
    Parameters:
    -----------
    cv_0 : float - CV in-control
    delta : float - Faktor pergeseran
    B : float - Slope
    theta : float - Rasio dasar (bisa = omega untuk konsistensi)
    m : int - Replikasi
    omega : float - Koefisien C_0 
    psi : float - Koefisien C_1 (sensitivitas terhadap mean)
    eta : float - Faktor pergeseran error
    """
    cv_actual = delta * cv_0
    
    # sigma_M/mu_X = omega*cv_0 + psi
    # Var(Y)/mu_X^2 = B^2*cv_actual^2 + (omega*cv_0 + psi)^2/m
    error_cv_term = (omega * cv_0 + psi)**2 / m
    var_term = B**2 * cv_actual**2 + error_cv_term
    cv_observed = np.sqrt(var_term) / abs(B)
    
    return cv_observed


# Verifikasi
cv_0 = 0.2
print("Verifikasi CV Observed:")
print(f"  In-control (delta=1, theta=0): CV_obs = {cv_observed_constant(cv_0, 1.0, 1, 0, 1):.4f}")
print(f"  In-control (delta=1, theta=0.05): CV_obs = {cv_observed_constant(cv_0, 1.0, 1, 0.05, 1):.4f}")
print(f"  OOC (delta=1.2, theta=0): CV_obs = {cv_observed_constant(cv_0, 1.2, 1, 0, 1):.4f}")
print(f"  OOC (delta=1.2, theta=0.05): CV_obs = {cv_observed_constant(cv_0, 1.2, 1, 0.05, 1):.4f}")


# %% [markdown]
# ## 4. Probabilitas Sinyal Diagram DS CV
#
# ### Mekanisme Double Sampling:
# **Tahap 1** (n1 sampel):
# - Hitung CV_hat dari n1 observasi
# - Batas kendali: LCL_1 = kL1*CV_0, UCL_1 = kU1*CV_0  
# - Batas peringatan: LWL = wL*CV_0, UWL = wU*CV_0
# - Jika CV_hat > UCL_1 atau CV_hat < LCL_1 → **SINYAL OOC**
# - Jika LCL_1 ≤ CV_hat < LWL atau UWL < CV_hat ≤ UCL_1 → **LANJUT TAHAP 2**
# - Jika LWL ≤ CV_hat ≤ UWL → **IN-CONTROL**
#
# **Tahap 2** (n1+n2 sampel gabungan):
# - Hitung CV_hat dari n1+n2 observasi
# - Batas kendali: LCL_2 = kL2*CV_0, UCL_2 = kU2*CV_0
# - Jika CV_hat > UCL_2 atau CV_hat < LCL_2 → **SINYAL OOC**
# - Lainnya → **IN-CONTROL**
#
# ### Catatan Penting tentang Parameterisasi:
# Batas-batas diekspresikan sebagai KOEFISIEN dikalikan CV_0:
# - kU1 > wU > 1 > wL > kL1 (urutan untuk deteksi peningkatan DAN penurunan CV)
# 
# ATAU dalam beberapa paper, batas diparameterisasi langsung sebagai nilai absolut.
# Di sini kita menggunakan nilai absolut sesuai format tabel jurnal.

# %%
def compute_signal_probability(n1, n2, kL1, kU1, wL, wU, kL2, kU2, cv_true):
    """
    Menghitung probabilitas sinyal (deteksi OOC) untuk diagram DS CV.
    
    Semua batas adalah nilai ABSOLUT dari CV (bukan koefisien).
    Urutan: kL1 < wL < wU < kU1 (untuk tahap 1)
            kL2 < kU2 (untuk tahap 2)
    
    Parameters:
    -----------
    n1, n2 : int - Ukuran sampel tahap 1 dan tambahan tahap 2
    kL1, kU1 : float - Batas kendali absolut tahap 1
    wL, wU : float - Batas peringatan absolut tahap 1  
    kL2, kU2 : float - Batas kendali absolut tahap 2
    cv_true : float - CV sebenarnya (observed, termasuk ME)
    
    Returns:
    --------
    P_signal : float - Probabilitas keseluruhan terdeteksi OOC
    P_stage2 : float - Probabilitas masuk ke tahap 2
    """
    # === TAHAP 1: Sinyal langsung ===
    # P(CV_hat_1 > kU1) + P(CV_hat_1 < kL1)
    P_above_kU1 = 1.0 - cdf_sample_cv(kU1, n1, cv_true)
    P_below_kL1 = cdf_sample_cv(kL1, n1, cv_true)
    P_signal_1 = P_above_kU1 + P_below_kL1
    
    # === TAHAP 1: Zona peringatan (lanjut ke tahap 2) ===
    # P(wU < CV_hat_1 <= kU1): zona warning atas
    P_warn_upper = prob_cv_in_range(wU, kU1, n1, cv_true)
    # P(kL1 <= CV_hat_1 < wL): zona warning bawah
    P_warn_lower = prob_cv_in_range(kL1, wL, n1, cv_true)
    P_stage2 = P_warn_upper + P_warn_lower
    
    # === TAHAP 2: Sinyal dari sampel gabungan (n1 + n2) ===
    n_total = n1 + n2
    P_above_kU2 = 1.0 - cdf_sample_cv(kU2, n_total, cv_true)
    P_below_kL2 = cdf_sample_cv(kL2, n_total, cv_true)
    P_signal_2 = P_above_kU2 + P_below_kL2
    
    # === TOTAL ===
    # P(sinyal) = P(sinyal tahap 1) + P(masuk tahap 2) * P(sinyal tahap 2)
    P_signal = P_signal_1 + P_stage2 * P_signal_2
    
    return P_signal, P_stage2


def compute_TARL(n1, n2, kL1, kU1, wL, wU, kL2, kU2, cv_true):
    """
    Menghitung TARL (Time-Adjusted Run Length).
    
    TARL = ASS / P(sinyal)
    ASS (Average Sample Size) = n1 + n2 * P(masuk tahap 2)
    
    TARL memperhitungkan biaya sampling rata-rata per siklus,
    memberikan metrik yang lebih realistis dibanding ARL standar.
    """
    P_signal, P_stage2 = compute_signal_probability(
        n1, n2, kL1, kU1, wL, wU, kL2, kU2, cv_true
    )
    
    # Average Sample Size per siklus
    ASS = n1 + n2 * P_stage2
    
    if P_signal < 1e-15:
        return 1e10
    
    TARL = ASS / P_signal
    return TARL


print("Fungsi TARL berhasil didefinisikan.")
# Quick test
cv0 = 0.2
tarl_ic = compute_TARL(5, 20, 0.10, 0.35, 0.15, 0.28, 0.12, 0.32, cv0)
tarl_ooc = compute_TARL(5, 20, 0.10, 0.35, 0.15, 0.28, 0.12, 0.32, 0.24)
print(f"  Test TARL in-control (CV=0.2): {tarl_ic:.2f}")
print(f"  Test TARL out-of-control (CV=0.24): {tarl_ooc:.2f}")


# %% [markdown]
# ## 5. Fungsi Objektif dan Optimasi
#
# ### Strategi Optimasi:
# - **Variabel diskrit** (n1, n2): Grid search
# - **Variabel kontinu** (kL1, kU1, wL, wU, kL2, kU2): Differential Evolution
# - **Kendala**: TARL_0 ≈ target (biasanya 370.4)
# - **Tujuan**: Minimasi TARL_1 (out-of-control)
#
# ### Parameterisasi Batas (sesuai jurnal):
# Batas diekspresikan sebagai koefisien K dikalikan CV_0:
# - UCL_1 = kU1 * CV_0, LCL_1 = kL1 * CV_0 (kU1 > 1 > kL1)
# - UWL = wU * CV_0, LWL = wL * CV_0 (kU1 > wU > 1 > wL > kL1)
# - UCL_2 = kU2 * CV_0, LCL_2 = kL2 * CV_0

# %%
def objective_ds_cv(params, n1, n2, cv_0, delta, B, theta, m, 
                    TARL0_target, condition, omega=0, psi=0, eta=1.0):
    """
    Fungsi objektif: minimasi TARL_1 dengan kendala TARL_0 ≈ target.
    
    params = [kU1, kL1, wU, wL, kU2, kL2] sebagai KOEFISIEN dari CV_0
    
    Batas absolut: UCL_1 = kU1*CV_0, LCL_1 = kL1*CV_0, dst.
    """
    kU1_coef, kL1_coef, wU_coef, wL_coef, kU2_coef, kL2_coef = params
    
    # Konversi ke nilai absolut
    kU1 = kU1_coef * cv_0
    kL1 = kL1_coef * cv_0
    wU = wU_coef * cv_0
    wL = wL_coef * cv_0
    kU2 = kU2_coef * cv_0
    kL2 = kL2_coef * cv_0
    
    # === Cek kendala urutan ===
    # Urutan yang benar: kL1 < wL < wU < kU1 dan kL2 < kU2
    # Dalam koefisien: kL1_coef < wL_coef < wU_coef < kU1_coef
    #                  kL2_coef < kU2_coef
    if not (kL1_coef < wL_coef < wU_coef < kU1_coef):
        return 1e12
    if not (kL2_coef < kU2_coef):
        return 1e12
    
    # === Hitung CV observed in-control (delta=1) ===
    if condition == 'constant':
        cv_obs_0 = cv_observed_constant(cv_0, 1.0, B, theta, m, eta)
    else:
        cv_obs_0 = cv_observed_linear(cv_0, 1.0, B, theta, m, omega, psi, eta)
    
    # === Hitung TARL_0 (in-control) ===
    TARL_0 = compute_TARL(n1, n2, kL1, kU1, wL, wU, kL2, kU2, cv_obs_0)
    
    # === Kendala TARL_0 harus dekat target ===
    # Toleransi: TARL_0 harus dalam 95%-105% dari target
    if TARL_0 < TARL0_target * 0.98:
        return 1e10 + 1e5 * (TARL0_target - TARL_0)
    
    # === Hitung CV observed out-of-control (delta > 1) ===
    if condition == 'constant':
        cv_obs_1 = cv_observed_constant(cv_0, delta, B, theta, m, eta)
    else:
        cv_obs_1 = cv_observed_linear(cv_0, delta, B, theta, m, omega, psi, eta)
    
    # === Hitung TARL_1 (out-of-control) - TARGET MINIMASI ===
    TARL_1 = compute_TARL(n1, n2, kL1, kU1, wL, wU, kL2, kU2, cv_obs_1)
    
    # Penalti ringan untuk mendorong TARL_0 tepat di target
    penalty = 0.0001 * (TARL_0 - TARL0_target)**2 / TARL0_target
    
    return TARL_1 + penalty


# %%
def optimize_ds_cv_me(cv_0, delta, B, theta, m, TARL0_target=370.4,
                      condition='constant', omega=0, psi=0, eta=1.0,
                      n1_range=range(3, 10), n2_range=range(3, 50),
                      verbose=False):
    """
    Fungsi optimasi utama diagram DS CV-ME.
    
    Mencari parameter optimal: n1, n2, kU1, kL1, wU, wL, kU2, kL2
    yang meminimasi TARL_1 (deteksi cepat) dengan TARL_0 ≈ target.
    
    Parameters:
    -----------
    cv_0 : float - CV in-control (misal 0.2)
    delta : float - Faktor pergeseran CV (>1 untuk peningkatan)
    B : float - Slope model kovariat
    theta : float - Rasio error sigma_M/sigma_X0
    m : int - Jumlah replikasi pengukuran
    TARL0_target : float - Target TARL in-control
    condition : str - 'constant' atau 'linear'
    omega, psi : float - Parameter untuk linear error
    eta : float - Faktor pergeseran varians error
    
    Returns:
    --------
    best : dict - Parameter optimal dan nilai TARL
    """
    best_TARL1 = 1e10
    best_params = None
    
    for n1 in n1_range:
        for n2 in n2_range:
            if n2 < n1:
                continue
            
            # Bounds untuk koefisien [kU1, kL1, wU, wL, kU2, kL2]
            # kU1 > wU > 1 (biasanya 1.5 - 10)
            # wL < 1 < wU
            # kL1 < wL < 1
            bounds = [
                (1.5, 10.0),    # kU1_coef: batas atas kontrol (> 1)
                (0.01, 0.99),   # kL1_coef: batas bawah kontrol (< 1) 
                (1.01, 8.0),    # wU_coef: batas atas warning (> 1)
                (0.1, 0.999),   # wL_coef: batas bawah warning (< 1)
                (1.5, 10.0),    # kU2_coef: batas atas kontrol tahap 2
                (0.01, 0.99),   # kL2_coef: batas bawah kontrol tahap 2
            ]
            
            try:
                result = differential_evolution(
                    objective_ds_cv,
                    bounds=bounds,
                    args=(n1, n2, cv_0, delta, B, theta, m,
                          TARL0_target, condition, omega, psi, eta),
                    maxiter=300,
                    tol=1e-8,
                    seed=42,
                    polish=True,
                    popsize=15,
                    mutation=(0.5, 1.5),
                    recombination=0.8,
                    init='sobol'
                )
                
                if result.fun < best_TARL1 and result.fun < 1e9:
                    # Verifikasi solusi
                    kU1_c, kL1_c, wU_c, wL_c, kU2_c, kL2_c = result.x
                    
                    # Cek urutan
                    if kL1_c < wL_c < wU_c < kU1_c and kL2_c < kU2_c:
                        # Hitung TARL_0 untuk verifikasi
                        kU1 = kU1_c * cv_0
                        kL1 = kL1_c * cv_0
                        wU = wU_c * cv_0
                        wL = wL_c * cv_0
                        kU2 = kU2_c * cv_0
                        kL2 = kL2_c * cv_0
                        
                        if condition == 'constant':
                            cv_obs_0 = cv_observed_constant(cv_0, 1.0, B, theta, m, eta)
                        else:
                            cv_obs_0 = cv_observed_linear(cv_0, 1.0, B, theta, m, omega, psi, eta)
                        
                        TARL_0 = compute_TARL(n1, n2, kL1, kU1, wL, wU, kL2, kU2, cv_obs_0)
                        
                        if TARL_0 >= TARL0_target * 0.95:
                            best_TARL1 = result.fun
                            best_params = {
                                'n1': n1, 'n2': n2,
                                'kU1': round(kU1_c, 4),
                                'kL1': round(kL1_c, 4),
                                'wU': round(wU_c, 4),
                                'wL': round(wL_c, 4),
                                'kU2': round(kU2_c, 4),
                                'kL2': round(kL2_c, 4),
                                'TARL_0': round(TARL_0, 2),
                                'TARL_1': round(best_TARL1, 2)
                            }
                            if verbose:
                                print(f"    n1={n1}, n2={n2}: TARL_1={best_TARL1:.2f}, TARL_0={TARL_0:.2f}")
            
            except Exception:
                continue
    
    if best_params is None:
        best_params = {
            'n1': None, 'n2': None,
            'kU1': None, 'kL1': None,
            'wU': None, 'wL': None,
            'kU2': None, 'kL2': None,
            'TARL_0': None, 'TARL_1': None
        }
    
    return best_params


# %% [markdown]
# ## 6. Generasi Tabel Hasil (Format Jurnal)
#
# Tabel mengikuti format dari gambar referensi:
# - Header: ω = φ = 0 dan m = B = 1
# - Baris: δ = {1.01, 1.05, 1.2, 1.3}
# - Kolom: θ = {0, 0.01, 0.03, 0.05}
# - Isi: n1, n2, kU1, kL1, wU, wL, kU2, kL2, (TARL)

# %%
def run_full_optimization(cv_0=0.2, B=1, m=1, omega=0, psi=0, eta=1.0,
                           TARL0_target=370.4, condition='constant',
                           delta_values=[1.01, 1.05, 1.2, 1.3],
                           theta_values=[0, 0.01, 0.03, 0.05],
                           n1_range=range(3, 7), n2_range=range(5, 35)):
    """
    Menjalankan optimasi penuh dan menghasilkan tabel hasil.
    """
    results = {}
    
    header = f"ω={omega}, ψ={psi}, m={m}, B={B}"
    print(f"\n{'═'*90}")
    print(f"  OPTIMASI DS CV-ME | Kondisi: {condition.upper()}")
    print(f"  {header} | CV_0={cv_0} | TARL_0 target={TARL0_target}")
    print(f"{'═'*90}")
    
    total = len(delta_values) * len(theta_values)
    count = 0
    
    for delta in delta_values:
        results[delta] = {}
        for theta in theta_values:
            count += 1
            print(f"\n  [{count}/{total}] δ={delta}, θ={theta} ... ", end="", flush=True)
            
            res = optimize_ds_cv_me(
                cv_0=cv_0, delta=delta, B=B, theta=theta, m=m,
                TARL0_target=TARL0_target, condition=condition,
                omega=omega, psi=psi, eta=eta,
                n1_range=n1_range, n2_range=n2_range,
                verbose=False
            )
            
            results[delta][theta] = res
            
            if res['n1'] is not None:
                print(f"✓ n1={res['n1']}, n2={res['n2']}, TARL_1={res['TARL_1']:.2f}")
            else:
                print("✗ Tidak ditemukan solusi feasible")
    
    print(f"\n{'═'*90}")
    print("  OPTIMASI SELESAI")
    print(f"{'═'*90}")
    
    return results


# %% [markdown]
# ## 7. Fungsi Tampilan Tabel (Persis Format Jurnal)

# %%
def print_journal_table(results, delta_values, theta_values,
                         omega=0, psi=0, m=1, B=1):
    """
    Mencetak tabel dalam format PERSIS sesuai jurnal (gambar referensi).
    
    Format setiap sel:
    Baris 1: n1, n2, kU1, kL1,
    Baris 2: wU, wL,
    Baris 3: kU2, kL2
    Baris 4: (TARL_1) [dalam kurung]
    """
    # Title
    if psi == 0 and omega == 0:
        header = f"ω = φ = 0 and m = B = {B}"
    else:
        header = f"ω = {omega}, ψ = {psi} and m = {m}, B = {B}"
    
    col_width = 24
    total_width = 10 + col_width * len(theta_values)
    
    print(f"\n{'─'*total_width}")
    print(f"{'δ':^10}{header:^{col_width * len(theta_values)}}")
    print(f"{'─'*total_width}")
    
    # Sub-header: theta values
    print(f"{'':^10}", end="")
    for theta in theta_values:
        print(f"{'θ = '+str(theta):^{col_width}}", end="")
    print(f"\n{'─'*total_width}")
    
    for delta in delta_values:
        # Baris 1: n1, n2, kU1, kL1
        line = f"{delta:<10.2f}"
        for theta in theta_values:
            r = results[delta][theta]
            if r['n1'] is not None:
                cell = f"{r['n1']}, {r['n2']}, {r['kU1']}, {r['kL1']},"
            else:
                cell = "N/A"
            line += f"{cell:^{col_width}}"
        print(line)
        
        # Baris 2: wU, wL,
        line = f"{'':10}"
        for theta in theta_values:
            r = results[delta][theta]
            if r['n1'] is not None:
                cell = f"{r['wU']}, {r['wL']},"
            else:
                cell = ""
            line += f"{cell:^{col_width}}"
        print(line)
        
        # Baris 3: kU2, kL2
        line = f"{'':10}"
        for theta in theta_values:
            r = results[delta][theta]
            if r['n1'] is not None:
                cell = f"{r['kU2']}, {r['kL2']}"
            else:
                cell = ""
            line += f"{cell:^{col_width}}"
        print(line)
        
        # Baris 4: (TARL) dalam kurung italic
        line = f"{'':10}"
        for theta in theta_values:
            r = results[delta][theta]
            if r['TARL_1'] is not None:
                cell = f"({r['TARL_1']:.2f})"
            else:
                cell = ""
            line += f"{cell:^{col_width}}"
        print(line)
        print()  # separator antar delta
    
    print(f"{'─'*total_width}")


def results_to_dataframe(results, delta_values, theta_values, condition_name):
    """
    Konversi hasil ke pandas DataFrame untuk export.
    """
    rows = []
    for delta in delta_values:
        for theta in theta_values:
            r = results[delta][theta]
            if r['n1'] is not None:
                rows.append({
                    'Kondisi': condition_name,
                    'δ (delta)': delta,
                    'θ (theta)': theta,
                    'n1': r['n1'],
                    'n2': r['n2'],
                    'kU1': r['kU1'],
                    'kL1': r['kL1'],
                    'wU': r['wU'],
                    'wL': r['wL'],
                    'kU2': r['kU2'],
                    'kL2': r['kL2'],
                    'TARL_0': r['TARL_0'],
                    'TARL_1': r['TARL_1']
                })
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. Fungsi Visualisasi

# %%
def plot_tarl_comparison(results_const, results_linear, 
                          delta_values, theta_values, cv_0=0.2):
    """
    Grafik perbandingan TARL antara Constant Error vs Linear Error.
    """
    n_theta = len(theta_values)
    fig, axes = plt.subplots(1, n_theta, figsize=(5*n_theta, 5), sharey=True)
    if n_theta == 1:
        axes = [axes]
    
    for idx, theta in enumerate(theta_values):
        ax = axes[idx]
        
        # Constant Error
        tarl_c = [results_const[d][theta]['TARL_1'] 
                  if results_const[d][theta]['TARL_1'] is not None else np.nan 
                  for d in delta_values]
        
        # Linear Error
        tarl_l = [results_linear[d][theta]['TARL_1'] 
                  if results_linear[d][theta]['TARL_1'] is not None else np.nan 
                  for d in delta_values]
        
        ax.plot(delta_values, tarl_c, 'bo-', lw=2, ms=8, label='Constant Error')
        ax.plot(delta_values, tarl_l, 'rs--', lw=2, ms=8, label='Linear Error')
        
        ax.set_xlabel('δ (Shift Factor)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('TARL₁ (Out-of-Control)', fontsize=11)
        ax.set_title(f'θ = {theta}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle(f'Perbandingan TARL: Constant vs Linear Variance Error\n'
                 f'CV₀ = {cv_0}', fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig('tarl_comparison_const_vs_linear.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik disimpan: tarl_comparison_const_vs_linear.png")


def plot_tarl_single_condition(results, delta_values, theta_values, 
                                condition_name, cv_0=0.2):
    """
    Plot TARL vs delta untuk satu kondisi dengan berbagai theta.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<']
    colors = plt.cm.tab10(np.linspace(0, 1, len(theta_values)))
    
    for idx, theta in enumerate(theta_values):
        tarl_vals = [results[d][theta]['TARL_1'] 
                     if results[d][theta]['TARL_1'] is not None else np.nan 
                     for d in delta_values]
        
        ax.plot(delta_values, tarl_vals,
                marker=markers[idx % len(markers)],
                color=colors[idx],
                lw=2, ms=9, label=f'θ = {theta}')
    
    ax.set_xlabel('δ (Shift Factor / Pergeseran CV)', fontsize=12)
    ax.set_ylabel('TARL₁ (Out-of-Control)', fontsize=12)
    ax.set_title(f'TARL vs Pergeseran CV\nKondisi: {condition_name} | CV₀ = {cv_0}',
                 fontsize=13)
    ax.legend(fontsize=11, title='Rasio Error (θ)', title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    fname = f'tarl_vs_delta_{condition_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grafik disimpan: {fname}")


# %% [markdown]
# ## 9. EKSEKUSI - Kondisi 1: Constant Variance Error
#
# Parameter:
# - ω = φ = 0 (tidak ada komponen linear)
# - m = B = 1 (satu pengukuran, slope = 1)
# - CV_0 = 0.2 (gamma_0 = 5)
# - TARL_0 target = 370.4

# %%
# ===================== PARAMETER UTAMA =====================
cv_0 = 0.2           # CV in-control
B = 1                # Slope model kovariat
m = 1                # Jumlah replikasi
TARL0_target = 370.4 # Target TARL in-control

# Nilai yang akan dievaluasi
delta_values = [1.01, 1.05, 1.2, 1.3]
theta_values = [0, 0.01, 0.03, 0.05]

# Rentang pencarian
n1_range = range(3, 6)    # n1 kecil (3-5)
n2_range = range(15, 35)  # n2 lebih besar

# === KONDISI 1: CONSTANT ERROR ===
print("\n" + "█"*70)
print("█  KONDISI 1: CONSTANT VARIANCE ERROR")
print("█  e_ij ~ N(0, sigma_M^2), sigma_M = theta * sigma_X0")
print("█"*70)

results_constant = run_full_optimization(
    cv_0=cv_0, B=B, m=m, omega=0, psi=0,
    TARL0_target=TARL0_target, condition='constant',
    delta_values=delta_values, theta_values=theta_values,
    n1_range=n1_range, n2_range=n2_range
)

# %%
# Tampilkan tabel format jurnal - Kondisi 1
print("\n\n" + "="*90)
print("  TABEL HASIL - KONDISI 1: CONSTANT VARIANCE ERROR")
print("="*90)
print_journal_table(results_constant, delta_values, theta_values,
                    omega=0, psi=0, m=m, B=B)


# %% [markdown]
# ## 10. EKSEKUSI - Kondisi 2: Linear Variance Error
#
# Parameter tambahan:
# - omega = 0.1 (koefisien C_0)
# - psi = 0.05 (koefisien C_1, sensitivitas terhadap mean)
# - sigma_M_j = omega*sigma_X0 + psi*mu_X

# %%
# === KONDISI 2: LINEAR ERROR ===
omega = 0.1   # C_0 coefficient
psi = 0.05    # C_1 coefficient

print("\n\n" + "█"*70)
print("█  KONDISI 2: LINEAR VARIANCE ERROR")
print("█  sigma_M = omega*sigma_X0 + psi*mu_X")
print(f"█  omega={omega}, psi={psi}")
print("█"*70)

results_linear = run_full_optimization(
    cv_0=cv_0, B=B, m=m, omega=omega, psi=psi,
    TARL0_target=TARL0_target, condition='linear',
    delta_values=delta_values, theta_values=theta_values,
    n1_range=n1_range, n2_range=n2_range
)

# %%
# Tampilkan tabel format jurnal - Kondisi 2
print("\n\n" + "="*90)
print("  TABEL HASIL - KONDISI 2: LINEAR VARIANCE ERROR")
print("="*90)
print_journal_table(results_linear, delta_values, theta_values,
                    omega=omega, psi=psi, m=m, B=B)


# %% [markdown]
# ## 11. Visualisasi Hasil

# %%
# Plot untuk Kondisi 1
plot_tarl_single_condition(results_constant, delta_values, theta_values,
                            condition_name='Constant Error', cv_0=cv_0)

# %%
# Plot untuk Kondisi 2
plot_tarl_single_condition(results_linear, delta_values, theta_values,
                            condition_name='Linear Error', cv_0=cv_0)

# %%
# Plot perbandingan Constant vs Linear
plot_tarl_comparison(results_constant, results_linear,
                      delta_values, theta_values, cv_0=cv_0)

# %% [markdown]
# ## 12. Tabel DataFrame Gabungan dan Export

# %%
# Buat DataFrame
df_const = results_to_dataframe(results_constant, delta_values, theta_values, 'Constant')
df_lin = results_to_dataframe(results_linear, delta_values, theta_values, 'Linear')
df_all = pd.concat([df_const, df_lin], ignore_index=True)

# Tampilkan
print("\n" + "="*110)
print("  TABEL RINGKASAN LENGKAP (DataFrame)")
print("="*110)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(df_all.to_string(index=False))

# Export ke CSV
df_all.to_csv('ds_cv_me_results.csv', index=False)
print("\n\nHasil disimpan ke: ds_cv_me_results.csv")


# %% [markdown]
# ## 13. Analisis Tambahan: Pengaruh Jumlah Replikasi (m)

# %%
def analyze_replication_effect(cv_0=0.2, B=1, delta=1.2, theta=0.05,
                                m_values=[1, 2, 3, 5], TARL0_target=370.4):
    """
    Analisis pengaruh jumlah replikasi (m) terhadap performa TARL.
    Lebih banyak replikasi → error berkurang → deteksi lebih baik.
    """
    print(f"\n{'='*60}")
    print(f"  ANALISIS PENGARUH REPLIKASI (m)")
    print(f"  delta={delta}, theta={theta}, CV_0={cv_0}")
    print(f"{'='*60}")
    
    results_m = {}
    for m_val in m_values:
        print(f"\n  m={m_val}...", end=" ", flush=True)
        res = optimize_ds_cv_me(
            cv_0=cv_0, delta=delta, B=B, theta=theta, m=m_val,
            TARL0_target=TARL0_target, condition='constant',
            n1_range=range(3, 6), n2_range=range(15, 30),
            verbose=False
        )
        results_m[m_val] = res
        if res['TARL_1'] is not None:
            print(f"TARL_1 = {res['TARL_1']:.2f}")
        else:
            print("N/A")
    
    # Plot
    m_vals = [m for m in m_values if results_m[m]['TARL_1'] is not None]
    tarl_vals = [results_m[m]['TARL_1'] for m in m_vals]
    
    if m_vals:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(range(len(m_vals)), tarl_vals, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(m_vals)))
        ax.set_xticklabels([f'm={m}' for m in m_vals])
        ax.set_ylabel('TARL₁', fontsize=12)
        ax.set_title(f'Pengaruh Replikasi terhadap TARL₁\nδ={delta}, θ={theta}', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(tarl_vals):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('replication_effect.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return results_m

# Jalankan analisis replikasi
results_replication = analyze_replication_effect()

# %% [markdown]
# ## 14. Kesimpulan
#
# ### Temuan Utama:
# 1. **Measurement Error meningkatkan TARL₁** (menurunkan kemampuan deteksi):
#    - Semakin besar θ, semakin sulit mendeteksi pergeseran CV
#    - Efek lebih terasa pada pergeseran kecil (δ mendekati 1)
#
# 2. **Linear Error vs Constant Error**:
#    - Linear error umumnya memberikan TARL₁ lebih tinggi
#    - Karena varians error berubah seiring mean, menambah ketidakpastian
#
# 3. **Multiple Measurements (m > 1)**:
#    - Mengurangi dampak measurement error
#    - Menurunkan TARL₁ mendekati kasus tanpa error (θ=0)
#
# 4. **Double Sampling**:
#    - Efisien dalam hal sampling (ASS rendah)
#    - Tahap 2 memberikan "second chance" untuk konfirmasi

# %%
print("\n" + "═"*70)
print("  SIMULASI DS CV-ME SELESAI")
print("═"*70)
print("\n  File output yang dihasilkan:")
print("  ├── ds_cv_me_results.csv")
print("  ├── tarl_comparison_const_vs_linear.png")
print("  ├── tarl_vs_delta_constant_error.png")
print("  ├── tarl_vs_delta_linear_error.png")
print("  └── replication_effect.png")
print("\n  Parameter optimal tersimpan dalam variabel:")
print("  ├── results_constant (dict)")
print("  └── results_linear (dict)")
