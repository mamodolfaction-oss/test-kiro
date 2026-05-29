"""
Optimasi Parameter DS CV-ME Chart — per Kombinasi (δ × Parameter)
Referensi: Nuriman et al. (2026) — JQMA
Model: Y_ij = A + B*X_ij + e_ij
Metrik: TARL (Time-Adjusted Run Length)
Optimasi: GA dijalankan TERPISAH untuk setiap sel tabel

VERSI MULTIPROCESSING:
- Menggunakan multiprocessing.Pool untuk looping tingkat tinggi
- Setiap worker menjalankan GA secara serial (tanpa nested parallelism)
- Cocok untuk komputasi berat dengan banyak kombinasi parameter
"""

import numpy as np
from scipy.stats import nct, ncf
from scipy.integrate import quad
import multiprocessing as mp
from functools import partial
import warnings
import time
import itertools
import os

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# PARAMETER GLOBAL
# ═══════════════════════════════════════════════════════════════════

GAMMA_0 = 0.05        # γ₀ : CV in-control
N0 = 5                # n₀ : target rata-rata ukuran sampel
H = 500               # H  : truncation TARL
DELTA_LIST = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]

N1_RANGE = [3, 4, 5]  # Kandidat n1
N_MAX = 31            # Batas maksimum n1+n2

# Bounds koefisien GA: [ku1, kl1, wu, wl, ku2, kl2]
DEFAULT_BOUNDS = [
    (1.0, 10.0),   # ku1
    (0.1, 5.0),    # kl1
    (0.1, 8.0),    # wu
    (0.1, 5.0),    # wl
    (0.5, 10.0),   # ku2
    (0.1, 5.0),    # kl2
]

POP_SIZE = 40
N_GENERATIONS = 60
PATIENCE = 15


# ═══════════════════════════════════════════════════════════════════
# MODEL γ* — Sesuai Paper Nuriman et al. (2026)
# ═══════════════════════════════════════════════════════════════════

def gamma0_star_const(gamma0, theta, eta, B, m):
    """γ*₀ in-control — Model A. Eq.(6)"""
    return gamma0 * np.sqrt(B**2 + eta**2 / m) / (theta + B)


def gamma_plus_star_const(delta, gamma0, theta, eta, B, m):
    """γ*₊ OOC — Model A. Eq.(8)"""
    return gamma0 * np.sqrt(B**2 * delta**2 + eta**2 / m) / (theta + B)


def gamma0_star_linear(gamma0, theta, omega, phi, B, m):
    """γ*₀ in-control — Model B. Eq.(13)"""
    return gamma0 * np.sqrt(B**2 + (omega**2 + phi**2) / m) / (theta + B)


def gamma_plus_star_linear(delta, gamma0, theta, omega, phi, B, m):
    """γ*₊ OOC — Model B. Eq.(14)"""
    num = gamma0 * np.sqrt(B**2 + (omega**2 + phi**2 / delta**2) / m)
    den = theta + B / delta
    return num / den


def get_gamma_stars(delta, gamma0, theta, B, m,
                    eta=0.0, omega=0.0, phi=0.0, error_model='constant'):
    """Kembalikan (γ*₀, γ*₊) sesuai model error."""
    if error_model == 'constant':
        g0s = gamma0_star_const(gamma0, theta, eta, B, m)
        gps = gamma_plus_star_const(delta, gamma0, theta, eta, B, m)
    else:
        g0s = gamma0_star_linear(gamma0, theta, omega, phi, B, m)
        gps = gamma_plus_star_linear(delta, gamma0, theta, omega, phi, B, m)
    return g0s, gps


# ═══════════════════════════════════════════════════════════════════
# DISTRIBUSI γ̂, BATAS KENDALI, DAN PROBABILITAS OOC
# ═══════════════════════════════════════════════════════════════════

def mean_std_gamma_hat(n_total, g):
    """μ dan σ dari γ̂ — Eq.(21)-(22) Hong et al. (2008)."""
    g2, g4, g6 = g**2, g**4, g**6
    N = n_total
    mu = g * (1
              + (1/N) * (g2 - 0.25)
              + (1/N**2) * (3*g4 - g2/4 - 7/32)
              + (1/N**3) * (15*g6 - 3*g4/4 - 7*g2/32 - 19/128))
    s2 = g**2 * ((1/N) * (g2 + 0.5)
                 + (1/N**2) * (8*g4 - g2 - 3/8)
                 + (1/N**3) * (69*g6 + 7*g4/2 + 3*g2/4 + 3/16))
    return mu, np.sqrt(max(s2, 1e-20))


def cdf_gamma_hat(x, n, g):
    """CDF γ̂ — Eq.(18)."""
    if x <= 0 or g <= 0:
        return 0.0
    return 1.0 - nct.cdf(np.sqrt(n) / x, df=n-1, nc=np.sqrt(n) / g)


def compute_limits(n1, n2, ku1, kl1, wu, wl, ku2, kl2, g0s):
    """Hitung 6 batas kendali dari koefisien — Eq.(20a)-(20d)."""
    mu1, s1 = mean_std_gamma_hat(n1, g0s)
    mu2, s2 = mean_std_gamma_hat(n1 + n2, g0s)
    return (
        mu1 + ku1 * s1,               # UCL₁
        max(mu1 - kl1 * s1, 1e-8),    # LCL₁
        mu1 + wu * s1,                 # UWL
        max(mu1 - wl * s1, 1e-8),     # LWL
        mu2 + ku2 * s2,               # UCL₂
        max(mu2 - kl2 * s2, 1e-8),    # LCL₂
    )


def is_valid_limits(UCL1, LCL1, UWL, LWL, UCL2, LCL2):
    """Validasi: UCL₁ > UWL > LWL > LCL₁ > 0, UCL₂ < UCL₁, LCL₂ > LCL₁."""
    return (UCL1 > UWL > LWL > LCL1 > 0
            and UCL2 < UCL1 and LCL2 > LCL1)


def compute_P_ooc(n1, n2, UCL1, LCL1, UWL, LWL, UCL2, LCL2, gps):
    """
    P^E_ooc = P_ooc1 + P_ooc2  [Eq.23]
    P_ooc1: langsung sinyal dari Tahap 1
    P_ooc2: masuk Tahap 2 → sinyal
    """
    # Stage 1 direct signal
    p_in = cdf_gamma_hat(UCL1, n1, gps) - cdf_gamma_hat(LCL1, n1, gps)
    p_ooc1 = max(1.0 - p_in, 0.0)

    # Stage 2 via integrasi numerik
    ncp1 = np.sqrt(n1) / gps
    sn = np.sqrt(n1)

    def integrand(u):
        g1 = u / sn
        if g1 <= 0:
            return 0.0
        f = (n1 - 1) / (n2 - 1) + 1
        u2 = u**2
        ua = max(f * UCL2**2 - (n1-1)*n1 / ((n2-1)*u2), 1e-12)
        la = max(f * LCL2**2 - (n1-1)*n1 / ((n2-1)*u2), 0.0)
        nc2 = n2 / (gps**2)
        pu = 1.0 - ncf.cdf(ua * n2/gps**2, dfn=1, dfd=n2-1, nc=nc2)
        pl = ncf.cdf(la * n2/gps**2, dfn=1, dfd=n2-1, nc=nc2)
        return max(pu + pl, 0.0) * nct.pdf(u, df=n1-1, nc=ncp1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v1, _ = quad(integrand, LCL1*sn, LWL*sn,
                         limit=80, epsabs=1e-4, epsrel=1e-2)
            v2, _ = quad(integrand, UWL*sn, UCL1*sn,
                         limit=80, epsabs=1e-4, epsrel=1e-2)
        p_ooc2 = (v1 + v2) if not (np.isnan(v1) or np.isnan(v2)) else 0.0
    except Exception:
        p_ooc2 = 0.0

    P_ooc = max(p_ooc1 + p_ooc2, 1e-12)
    return P_ooc, max(1.0 - P_ooc, 0.0)


# ═══════════════════════════════════════════════════════════════════
# TARL — Time-Adjusted Run Length
# ═══════════════════════════════════════════════════════════════════

def tarl_from_beta(beta, H_val=H):
    """TARL dari β (probabilitas IC per siklus) dengan truncation H."""
    if abs(beta - 1.0) < 1e-14:
        return float(H_val)
    if abs(beta) < 1e-14:
        return 1.0
    j_arr = np.arange(1, H_val)
    P_ooc = 1.0 - beta
    tarl = (np.sum(j_arr * beta**(j_arr - 1)) * P_ooc
            + H_val * beta**(H_val - 1))
    return max(float(tarl), 1.0)


def compute_TARL(delta, n1, n2, ku1, kl1, wu, wl, ku2, kl2,
                 gamma0=GAMMA_0, theta=0.0, B=1, m=1,
                 eta=0.0, omega=0.0, phi=0.0,
                 error_model='constant', H_val=H):
    """Hitung TARL(δ) untuk satu set parameter chart."""
    g0s_ic, _ = get_gamma_stars(1.0, gamma0, theta, B, m,
                                eta=eta, omega=omega, phi=phi,
                                error_model=error_model)
    _, gps = get_gamma_stars(delta, gamma0, theta, B, m,
                             eta=eta, omega=omega, phi=phi,
                             error_model=error_model)
    limits = compute_limits(n1, n2, ku1, kl1, wu, wl, ku2, kl2, g0s_ic)
    if not is_valid_limits(*limits):
        return float(H_val)
    _, beta = compute_P_ooc(n1, n2, *limits, gps)
    return tarl_from_beta(beta, H_val)


# ═══════════════════════════════════════════════════════════════════
# GENETIC ALGORITHM — Serial per Worker
# ═══════════════════════════════════════════════════════════════════

def _fitness_sel(chromosome, n1, n2, delta, gamma0, theta, B, m,
                 eta, omega, phi, error_model, H_val):
    """Fitness function GA: minimasi TARL(δ)."""
    ku1, kl1, wu, wl, ku2, kl2 = chromosome
    g0s_ic, _ = get_gamma_stars(1.0, gamma0, theta, B, m,
                                eta=eta, omega=omega, phi=phi,
                                error_model=error_model)
    _, gps = get_gamma_stars(delta, gamma0, theta, B, m,
                             eta=eta, omega=omega, phi=phi,
                             error_model=error_model)
    limits = compute_limits(n1, n2, ku1, kl1, wu, wl, ku2, kl2, g0s_ic)
    if not is_valid_limits(*limits):
        return 1e6
    _, beta = compute_P_ooc(n1, n2, *limits, gps)
    return tarl_from_beta(beta, H_val)


def _ga_satu(n1, n2, delta, gamma0, theta, B, m,
             eta, omega, phi, error_model, H_val,
             bounds, pop_size, n_gen, patience, seed):
    """GA untuk SATU kombinasi (n1, n2, delta, parameter). Serial."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    pop = [list(rng.uniform(lo, hi)) for _ in range(pop_size)]
    best_chrom, best_fit, no_imp = None, 1e9, 0

    for _ in range(n_gen):
        fits = [_fitness_sel(ch, n1, n2, delta, gamma0, theta, B, m,
                             eta, omega, phi, error_model, H_val)
                for ch in pop]
        bi = int(np.argmin(fits))
        if fits[bi] < best_fit - 1e-7:
            best_fit, best_chrom, no_imp = fits[bi], pop[bi][:], 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

        # Tournament selection (k=3)
        n = len(pop)
        sel = []
        for _ in range(n):
            cands = rng.choice(n, size=3, replace=False)
            sel.append(pop[min(cands, key=lambda i: fits[i])][:])

        # BLX-0.5 crossover
        offspring = []
        for i in range(0, n - 1, 2):
            if rng.random() < 0.85:
                c1, c2 = [], []
                for g1, g2 in zip(sel[i], sel[i+1]):
                    span = abs(g2 - g1)
                    ext = 0.5 * span
                    lo_idx, hi_idx = len(c1), len(c2)
                    c1.append(float(np.clip(
                        rng.uniform(min(g1, g2) - ext, max(g1, g2) + ext),
                        lo[lo_idx], hi[lo_idx])))
                    c2.append(float(np.clip(
                        rng.uniform(min(g1, g2) - ext, max(g1, g2) + ext),
                        lo[hi_idx], hi[hi_idx])))
                offspring.extend([c1, c2])
            else:
                offspring.extend([sel[i][:], sel[i+1][:]])

        # Gaussian mutation (rate=15%)
        for j in range(len(offspring)):
            for k_idx in range(len(bounds)):
                if rng.random() < 0.15:
                    offspring[j][k_idx] = float(np.clip(
                        offspring[j][k_idx] + rng.normal(0, 0.1*(hi[k_idx]-lo[k_idx])),
                        lo[k_idx], hi[k_idx]))

        if best_chrom is not None:
            offspring[0] = best_chrom[:]
        pop = offspring[:pop_size]

    return best_chrom, best_fit


# ═══════════════════════════════════════════════════════════════════
# WORKER FUNCTIONS untuk multiprocessing.Pool
# ═══════════════════════════════════════════════════════════════════

def _worker_optimasi_n1n2(args):
    """
    Worker level 2: optimasi satu kombinasi (n1, n2) untuk satu sel.
    Dipanggil secara serial di dalam _worker_satu_sel.
    """
    (n1, n2, delta, gamma0, theta, B, m,
     eta, omega, phi, error_model, H_val,
     bounds, pop_size, n_gen, patience, seed) = args

    warnings.filterwarnings("ignore")
    chrom, fit = _ga_satu(n1, n2, delta, gamma0, theta, B, m,
                          eta, omega, phi, error_model, H_val,
                          bounds, pop_size, n_gen, patience, seed)
    return {'n1': n1, 'n2': n2, 'chrom': chrom, 'tarl': fit}


def _worker_satu_sel(args):
    """
    Worker level 1 (TOP-LEVEL PARALLELISM):
    Optimasi satu sel tabel (delta × val_param).
    Iterasi semua (n1, n2) secara SERIAL di dalam worker ini.
    Mengembalikan hasil terbaik untuk sel ini.
    """
    (delta, val_param, nama_param, base_params, error_model,
     n1_range, N_max, H_val, bounds, pop_size, n_gen, patience) = args

    warnings.filterwarnings("ignore")

    # Siapkan parameter lengkap
    p = dict(base_params)
    p[nama_param] = val_param

    gamma0 = p.get('gamma0', GAMMA_0)
    theta = p.get('theta', 0.0)
    B = p.get('B', 1)
    m = p.get('m', 1)
    eta = p.get('eta', 0.0)
    omega = p.get('omega', 0.0)
    phi = p.get('phi', 0.0)

    # Semua kombinasi (n1, n2)
    kombinasi = [(n1, n2)
                 for n1 in n1_range
                 for n2 in range(N0, N_max + 1)
                 if n1 + n2 <= N_max]

    best_result = None

    for n1, n2 in kombinasi:
        seed = 42 + n1 * 100 + n2
        chrom, fit = _ga_satu(n1, n2, delta, gamma0, theta, B, m,
                              eta, omega, phi, error_model, H_val,
                              bounds, pop_size, n_gen, patience, seed)
        if chrom is not None:
            if best_result is None or fit < best_result['tarl']:
                best_result = {
                    'n1': n1, 'n2': n2,
                    'chrom': chrom, 'tarl': fit
                }

    if best_result is not None:
        best_result['koef'] = {
            k: v for k, v in
            zip(['ku1', 'kl1', 'wu', 'wl', 'ku2', 'kl2'], best_result['chrom'])
        }

    return (delta, val_param, best_result)


# ═══════════════════════════════════════════════════════════════════
# FUNGSI UTAMA: buat_tabel_jurnal — MULTIPROCESSING VERSION
# ═══════════════════════════════════════════════════════════════════

def buat_tabel_jurnal(
        variasi_param,
        nama_param,
        base_params,
        error_model='constant',
        delta_list=None,
        n1_range=None,
        N_max=N_MAX,
        H_val=H,
        bounds=None,
        pop_size=POP_SIZE,
        n_gen=N_GENERATIONS,
        patience=PATIENCE,
        n_workers=None,
        verbose=True,
):
    """
    Buat tabel di mana SETIAP sel (delta × nilai_param) dioptimasi INDEPENDEN
    menggunakan multiprocessing.Pool.

    Strategi parallelism:
    - Level TINGGI: setiap sel tabel = 1 task di Pool
    - Level RENDAH (dalam worker): loop (n1,n2) serial
    - Ini optimal karena jumlah sel >> jumlah core,
      dan setiap sel cukup berat untuk amortize overhead spawn.

    Args:
        n_workers: jumlah proses paralel (None = cpu_count)
    """
    if delta_list is None:
        delta_list = DELTA_LIST
    if n1_range is None:
        n1_range = N1_RANGE
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    if n_workers is None:
        n_workers = mp.cpu_count()

    semua_sel = list(itertools.product(delta_list, variasi_param))

    if verbose:
        tetap = {k: v for k, v in base_params.items() if k != nama_param}
        print(f"\n{'═'*65}")
        print(f"  OPTIMASI: variasi {nama_param} | model: {error_model}")
        print(f"  Parameter tetap: " +
              ", ".join(f"{k}={v}" for k, v in tetap.items()))
        print(f"  {len(semua_sel)} sel = "
              f"{len(delta_list)} δ × {len(variasi_param)} kolom")
        print(f"  Workers: {n_workers} proses (multiprocessing.Pool)")
        print(f"{'═'*65}")

    t0 = time.time()

    # Siapkan argumen untuk setiap worker
    task_args = [
        (delta, val, nama_param, base_params, error_model,
         n1_range, N_max, H_val, bounds, pop_size, n_gen, patience)
        for delta, val in semua_sel
    ]

    # MULTIPROCESSING POOL — paralel di level sel tabel
    with mp.Pool(processes=n_workers) as pool:
        results_flat = pool.map(_worker_satu_sel, task_args)

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✅ Selesai! Waktu: {elapsed/60:.1f} menit\n")

    # Susun hasil: hasil[delta][val_param]
    hasil = {}
    for (delta, val_param, res) in results_flat:
        hasil.setdefault(delta, {})[val_param] = res

    return hasil


# ═══════════════════════════════════════════════════════════════════
# CETAK TABEL FORMAT JURNAL JQMA
# ═══════════════════════════════════════════════════════════════════

def cetak_tabel_jurnal(hasil, variasi_param, nama_param,
                       delta_list=None, judul=""):
    """Cetak tabel persis format paper JQMA."""
    if delta_list is None:
        delta_list = sorted(hasil.keys())

    lebar = 26
    n_col = len(variasi_param)
    sep = "─" * (10 + lebar * n_col)

    if judul:
        print(f"\n  {judul}")
    print(sep)
    header = f"{'δ':^10}" + "".join(
        f"{f'{nama_param}={v}':^{lebar}}" for v in variasi_param)
    print(header)
    print(sep)

    for delta in delta_list:
        baris = [["", "", "", ""] for _ in variasi_param]
        for ci, val in enumerate(variasi_param):
            res = hasil.get(delta, {}).get(val)
            if res is None or res.get('chrom') is None:
                baris[ci] = ["N/A", "", "", ""]
                continue
            n1, n2 = res['n1'], res['n2']
            k = res['koef']
            tarl_v = res['tarl']
            baris[ci] = [
                f"{n1}, {n2}, {k['ku1']:.4f}, {k['kl1']:.4f},",
                f"{k['wu']:.4f}, {k['wl']:.4f},",
                f"{k['ku2']:.4f}, {k['kl2']:.4f}",
                f"({tarl_v:.2f})",
            ]

        prefix = [f"{delta:<10.4f}", " "*10, " "*10, " "*10]
        for row_idx in range(4):
            line = prefix[row_idx]
            for ci in range(n_col):
                line += f"{baris[ci][row_idx]:^{lebar}}"
            print(line)
        print()

    print(sep)


def cetak_tabel_df(hasil, variasi_param, nama_param, delta_list=None):
    """DataFrame pandas dari hasil optimasi."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas tidak tersedia")
        return None

    if delta_list is None:
        delta_list = sorted(hasil.keys())

    rows = []
    for delta in delta_list:
        for val in variasi_param:
            res = hasil.get(delta, {}).get(val)
            if res is None or res.get('chrom') is None:
                continue
            k = res['koef']
            rows.append({
                'delta': delta,
                nama_param: val,
                'n1': res['n1'],
                'n2': res['n2'],
                'ku1': round(k['ku1'], 4),
                'kl1': round(k['kl1'], 4),
                'wu': round(k['wu'], 4),
                'wl': round(k['wl'], 4),
                'ku2': round(k['ku2'], 4),
                'kl2': round(k['kl2'], 4),
                'TARL': round(res['tarl'], 4),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# VISUALISASI
# ═══════════════════════════════════════════════════════════════════

def plot_perbandingan_tarl(hasil_dict, delta_list=None, judul=""):
    """Plot TARL vs delta untuk semua variasi parameter."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if delta_list is None:
        delta_list = DELTA_LIST

    n_plot = len(hasil_dict)
    fig, axes = plt.subplots(1, n_plot, figsize=(5*n_plot, 5), sharey=False)
    if n_plot == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, 6))

    for ax_idx, (label, (hasil, variasi_param, nama_param)) in enumerate(
            hasil_dict.items()):
        ax = axes[ax_idx]
        for ci, val in enumerate(variasi_param):
            tarl_vals, delta_ok = [], []
            for delta in delta_list:
                res = hasil.get(delta, {}).get(val)
                if res and res.get('chrom'):
                    tarl_vals.append(res['tarl'])
                    delta_ok.append(delta)
            if tarl_vals:
                ax.plot(delta_ok, tarl_vals, 'o-',
                        color=colors[ci % len(colors)],
                        lw=2, ms=6,
                        label=f'{nama_param}={val}')

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('δ (Shift)', fontsize=10)
        ax.set_ylabel('TARL', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    if judul:
        plt.suptitle(judul, fontsize=13, y=1.02)
    plt.tight_layout()
    fname = 'tarl_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafik disimpan: {fname}")


# ═══════════════════════════════════════════════════════════════════
# SIMPAN HASIL KE CSV
# ═══════════════════════════════════════════════════════════════════

def simpan_semua(semua_hasil, output_dir='.'):
    """Simpan semua hasil ke CSV."""
    os.makedirs(output_dir, exist_ok=True)
    for fname, (hasil, variasi_param, nama_param) in semua_hasil.items():
        df = cetak_tabel_df(hasil, variasi_param, nama_param)
        if df is not None:
            path = os.path.join(output_dir, f"{fname}.csv")
            df.to_csv(path, index=False)
            print(f"  ✓ {path}  ({len(df)} baris)")


# ═══════════════════════════════════════════════════════════════════
# MAIN — Eksekusi semua tabel
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Wajib untuk multiprocessing di Windows/macOS
    mp.freeze_support()

    print("=" * 65)
    print("  DS CV-ME Chart Optimization — MULTIPROCESSING VERSION")
    print(f"  CPU cores tersedia: {mp.cpu_count()}")
    print("=" * 65)

    theta_list = [0, 0.01, 0.03, 0.05]
    eta_list = [0, 0.1, 0.3, 0.5]
    m_list = [1, 3, 5, 7]
    B_list = [1, 2, 3, 4]
    omega_list = [0.0, 0.1, 0.2, 0.3]
    phi_list = [0.0, 0.1, 0.2, 0.3]

    # ─── TABEL 1: Model A ───────────────────────────────────────
    print("\n▶ TABEL 1 — Model A: Variasi θ | η=0.28, m=B=1")
    hasil_A_theta = buat_tabel_jurnal(
        variasi_param=theta_list,
        nama_param='theta',
        base_params={'gamma0': GAMMA_0, 'theta': 0.0,
                     'eta': 0.28, 'B': 1, 'm': 1},
        error_model='constant',
    )
    cetak_tabel_jurnal(hasil_A_theta, theta_list, 'θ',
                       judul="Model A | η=0.28, m=B=1 | variasi θ")

    print("\n▶ TABEL 1 — Model A: Variasi η | θ=0, m=B=1")
    hasil_A_eta = buat_tabel_jurnal(
        variasi_param=eta_list,
        nama_param='eta',
        base_params={'gamma0': GAMMA_0, 'theta': 0.0,
                     'eta': 0.0, 'B': 1, 'm': 1},
        error_model='constant',
    )
    cetak_tabel_jurnal(hasil_A_eta, eta_list, 'η',
                       judul="Model A | θ=0, m=B=1 | variasi η")

    print("\n▶ TABEL 1 — Model A: Variasi m | θ=0.05, η=0.28, B=1")
    hasil_A_m = buat_tabel_jurnal(
        variasi_param=m_list,
        nama_param='m',
        base_params={'gamma0': GAMMA_0, 'theta': 0.05,
                     'eta': 0.28, 'B': 1, 'm': 1},
        error_model='constant',
    )
    cetak_tabel_jurnal(hasil_A_m, m_list, 'm',
                       judul="Model A | θ=0.05, η=0.28, B=1 | variasi m")

    print("\n▶ TABEL 1 — Model A: Variasi B | θ=0.05, η=0.28, m=1")
    hasil_A_B = buat_tabel_jurnal(
        variasi_param=B_list,
        nama_param='B',
        base_params={'gamma0': GAMMA_0, 'theta': 0.05,
                     'eta': 0.28, 'B': 1, 'm': 1},
        error_model='constant',
    )
    cetak_tabel_jurnal(hasil_A_B, B_list, 'B',
                       judul="Model A | θ=0.05, η=0.28, m=1 | variasi B")


    # ─── TABEL 3: Model B ───────────────────────────────────────
    print("\n▶ TABEL 3 — Model B: Variasi θ | ω=φ=0, m=B=1")
    hasil_B_theta = buat_tabel_jurnal(
        variasi_param=theta_list,
        nama_param='theta',
        base_params={'gamma0': GAMMA_0, 'theta': 0.0,
                     'omega': 0.0, 'phi': 0.0, 'B': 1, 'm': 1},
        error_model='linear',
    )
    cetak_tabel_jurnal(hasil_B_theta, theta_list, 'θ',
                       judul="Model B | ω=φ=0, m=B=1 | variasi θ")

    print("\n▶ TABEL 3 — Model B: Variasi ω | θ=φ=0, m=B=1")
    hasil_B_omega = buat_tabel_jurnal(
        variasi_param=omega_list,
        nama_param='omega',
        base_params={'gamma0': GAMMA_0, 'theta': 0.0,
                     'omega': 0.0, 'phi': 0.0, 'B': 1, 'm': 1},
        error_model='linear',
    )
    cetak_tabel_jurnal(hasil_B_omega, omega_list, 'ω',
                       judul="Model B | θ=φ=0, m=B=1 | variasi ω")

    print("\n▶ TABEL 3 — Model B: Variasi φ | θ=ω=0, m=B=1")
    hasil_B_phi = buat_tabel_jurnal(
        variasi_param=phi_list,
        nama_param='phi',
        base_params={'gamma0': GAMMA_0, 'theta': 0.0,
                     'omega': 0.0, 'phi': 0.0, 'B': 1, 'm': 1},
        error_model='linear',
    )
    cetak_tabel_jurnal(hasil_B_phi, phi_list, 'φ',
                       judul="Model B | θ=ω=0, m=B=1 | variasi φ")

    print("\n▶ TABEL 3 — Model B: Variasi m | θ=0.05, ω=φ=0.1, B=1")
    hasil_B_m = buat_tabel_jurnal(
        variasi_param=m_list,
        nama_param='m',
        base_params={'gamma0': GAMMA_0, 'theta': 0.05,
                     'omega': 0.1, 'phi': 0.1, 'B': 1, 'm': 1},
        error_model='linear',
    )
    cetak_tabel_jurnal(hasil_B_m, m_list, 'm',
                       judul="Model B | θ=0.05, ω=φ=0.1, B=1 | variasi m")

    print("\n▶ TABEL 3 — Model B: Variasi B | θ=0.05, ω=φ=0.1, m=1")
    hasil_B_B = buat_tabel_jurnal(
        variasi_param=B_list,
        nama_param='B',
        base_params={'gamma0': GAMMA_0, 'theta': 0.05,
                     'omega': 0.1, 'phi': 0.1, 'B': 1, 'm': 1},
        error_model='linear',
    )
    cetak_tabel_jurnal(hasil_B_B, B_list, 'B',
                       judul="Model B | θ=0.05, ω=φ=0.1, m=1 | variasi B")


    # ─── VISUALISASI ────────────────────────────────────────────
    print("\n▶ Membuat grafik perbandingan...")
    plot_perbandingan_tarl(
        {
            'Model A — variasi θ': (hasil_A_theta, theta_list, 'θ'),
            'Model B — variasi θ': (hasil_B_theta, theta_list, 'θ'),
        },
        judul="Perbandingan TARL: Model A vs Model B",
    )

    # ─── SIMPAN CSV ─────────────────────────────────────────────
    print("\n▶ Menyimpan semua hasil ke CSV...")
    simpan_semua({
        'tabel1_A_variasi_theta': (hasil_A_theta, theta_list, 'theta'),
        'tabel1_A_variasi_eta': (hasil_A_eta, eta_list, 'eta'),
        'tabel1_A_variasi_m': (hasil_A_m, m_list, 'm'),
        'tabel1_A_variasi_B': (hasil_A_B, B_list, 'B'),
        'tabel3_B_variasi_theta': (hasil_B_theta, theta_list, 'theta'),
        'tabel3_B_variasi_omega': (hasil_B_omega, omega_list, 'omega'),
        'tabel3_B_variasi_phi': (hasil_B_phi, phi_list, 'phi'),
        'tabel3_B_variasi_m': (hasil_B_m, m_list, 'm'),
        'tabel3_B_variasi_B': (hasil_B_B, B_list, 'B'),
    })

    print("\n" + "=" * 65)
    print("  SEMUA SELESAI!")
    print("=" * 65)
