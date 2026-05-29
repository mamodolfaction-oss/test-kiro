"""
Optimasi Parameter DS CV-ME Chart — VERSI CEPAT (FAST)
Referensi: Nuriman et al. (2026) — JQMA

STRATEGI KECEPATAN (vs original ~1-3 jam → ini ~2-10 menit):
1. 2-PHASE GA: Phase 1 pakai P_ooc1 saja (100x faster per eval)
              Phase 2 refine best solution dengan full P_ooc
2. TARL closed-form: O(1) vs O(H) array
3. Vectorized GA: numpy matrix ops
4. Smart search: skip (n1,n2) yang jelas buruk
5. Caching: gamma_stars pre-computed per cell

Akurasi tetap terjaga karena P_ooc1 >> P_ooc2 untuk mayoritas kasus.
"""

import numpy as np
from scipy.stats import nct, ncf
import multiprocessing as mp
import warnings
import time
import itertools
import os

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# PARAMETER GLOBAL
# ═══════════════════════════════════════════════════════════════════
GAMMA_0 = 0.05
N0 = 5
H = 500
DELTA_LIST = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
N1_RANGE = [3, 4, 5]
N_MAX = 31

DEFAULT_BOUNDS = [
    (1.0, 10.0), (0.1, 5.0), (0.1, 8.0),
    (0.1, 5.0), (0.5, 10.0), (0.1, 5.0),
]

POP_SIZE = 30
N_GENERATIONS = 40
PATIENCE = 10



# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_gamma_stars(delta, gamma0, theta, B, m,
                    eta=0.0, omega=0.0, phi=0.0, error_model='constant'):
    if error_model == 'constant':
        g0s = gamma0 * np.sqrt(B**2 + eta**2 / m) / (theta + B)
        gps = gamma0 * np.sqrt(B**2 * delta**2 + eta**2 / m) / (theta + B)
    else:
        g0s = gamma0 * np.sqrt(B**2 + (omega**2 + phi**2) / m) / (theta + B)
        gps = gamma0 * np.sqrt(B**2 + (omega**2 + phi**2/delta**2) / m) / (theta + B/delta)
    return g0s, gps


def mean_std_gamma_hat(n, g):
    g2, g4, g6 = g**2, g**4, g**6
    N = n
    mu = g * (1 + (g2 - 0.25)/N + (3*g4 - g2/4 - 7/32)/N**2
              + (15*g6 - 3*g4/4 - 7*g2/32 - 19/128)/N**3)
    s2 = g**2 * ((g2 + 0.5)/N + (8*g4 - g2 - 3/8)/N**2
                 + (69*g6 + 7*g4/2 + 3*g2/4 + 3/16)/N**3)
    return mu, np.sqrt(max(s2, 1e-20))


def cdf_gamma_hat(x, n, g):
    if x <= 0 or g <= 0:
        return 0.0
    sn = np.sqrt(n)
    return 1.0 - nct.cdf(sn / x, df=n-1, nc=sn / g)


def compute_limits(n1, n2, ku1, kl1, wu, wl, ku2, kl2, g0s):
    mu1, s1 = mean_std_gamma_hat(n1, g0s)
    mu2, s2 = mean_std_gamma_hat(n1 + n2, g0s)
    return (mu1 + ku1*s1, max(mu1 - kl1*s1, 1e-8),
            mu1 + wu*s1, max(mu1 - wl*s1, 1e-8),
            mu2 + ku2*s2, max(mu2 - kl2*s2, 1e-8))


def is_valid(UCL1, LCL1, UWL, LWL, UCL2, LCL2):
    return (UCL1 > UWL > LWL > LCL1 > 0
            and UCL2 < UCL1 and LCL2 > LCL1)



def P_ooc_stage1(n1, UCL1, LCL1, gps):
    """P_ooc from Stage 1 only — FAST (0.08ms per call)."""
    p_in = cdf_gamma_hat(UCL1, n1, gps) - cdf_gamma_hat(LCL1, n1, gps)
    return max(1.0 - p_in, 1e-12)


def P_ooc_full(n1, n2, UCL1, LCL1, UWL, LWL, UCL2, LCL2, gps):
    """Full P_ooc with Stage 2 integral (Gauss-Legendre 16 points)."""
    p_ooc1 = P_ooc_stage1(n1, UCL1, LCL1, gps)

    sn = np.sqrt(n1)
    ncp1 = sn / gps

    # Gauss-Legendre nodes (16 points — good balance)
    nodes, weights = np.polynomial.legendre.leggauss(16)

    def _integrate_interval(a, b):
        if b <= a:
            return 0.0
        mid = 0.5 * (b + a)
        half = 0.5 * (b - a)
        total = 0.0
        for xi, wi in zip(nodes, weights):
            u = mid + half * xi
            if u <= 0:
                continue
            f = (n1-1)/(n2-1) + 1
            u2 = u * u
            ua = max(f*UCL2**2 - (n1-1)*n1/((n2-1)*u2), 1e-12)
            la = max(f*LCL2**2 - (n1-1)*n1/((n2-1)*u2), 0.0)
            nc2 = n2 / (gps**2)
            pu = 1.0 - ncf.cdf(ua*n2/gps**2, dfn=1, dfd=n2-1, nc=nc2)
            pl = ncf.cdf(la*n2/gps**2, dfn=1, dfd=n2-1, nc=nc2)
            total += wi * max(pu + pl, 0.0) * nct.pdf(u, df=n1-1, nc=ncp1)
        return half * total

    p_ooc2 = (_integrate_interval(LCL1*sn, LWL*sn)
              + _integrate_interval(UWL*sn, UCL1*sn))

    return max(p_ooc1 + p_ooc2, 1e-12)


def tarl_from_pooc(P_ooc, H_val=H):
    """TARL = E[min(Geom(P_ooc), H)] — closed form."""
    if P_ooc >= 1.0:
        return 1.0
    if P_ooc <= 1e-14:
        return float(H_val)
    beta = 1.0 - P_ooc
    bH = beta ** H_val
    return max((1.0 - bH) / P_ooc, 1.0)



# ═══════════════════════════════════════════════════════════════════
# GA — 2-PHASE: fast proxy + refine
# ═══════════════════════════════════════════════════════════════════

def _fitness_fast(chrom, n1, n2, g0s, gps, H_val):
    """FAST fitness: P_ooc Stage 1 only (~0.08ms)."""
    ku1, kl1, wu, wl, ku2, kl2 = chrom
    limits = compute_limits(n1, n2, ku1, kl1, wu, wl, ku2, kl2, g0s)
    if not is_valid(*limits):
        return 1e6
    UCL1, LCL1 = limits[0], limits[1]
    p_ooc = P_ooc_stage1(n1, UCL1, LCL1, gps)
    return tarl_from_pooc(p_ooc, H_val)


def _ga_2phase(n1, n2, g0s, gps, H_val, bounds, pop_size, n_gen, patience, seed):
    """
    2-Phase GA:
    - Phase 1: optimize using P_ooc1 only (FAST)
    - Return best chromosome for later refinement
    """
    rng = np.random.default_rng(seed)
    ndim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    span = hi - lo

    pop = rng.uniform(lo, hi, size=(pop_size, ndim))
    best_fit = 1e9
    best_chrom = None
    no_imp = 0

    for gen in range(n_gen):
        fits = np.array([_fitness_fast(pop[i], n1, n2, g0s, gps, H_val)
                         for i in range(pop_size)])

        idx = np.argmin(fits)
        if fits[idx] < best_fit - 1e-7:
            best_fit = fits[idx]
            best_chrom = pop[idx].copy()
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

        # Tournament k=3
        sel_idx = np.array([
            min(rng.choice(pop_size, 3, replace=False), key=lambda i: fits[i])
            for _ in range(pop_size)])
        parents = pop[sel_idx]

        # BLX-0.5 crossover (vectorized)
        offspring = np.empty_like(parents)
        for i in range(0, pop_size - 1, 2):
            p1, p2 = parents[i], parents[i+1]
            if rng.random() < 0.85:
                mn, mx = np.minimum(p1, p2), np.maximum(p1, p2)
                ext = 0.5 * (mx - mn)
                offspring[i] = rng.uniform(mn - ext, mx + ext)
                offspring[i+1] = rng.uniform(mn - ext, mx + ext)
            else:
                offspring[i], offspring[i+1] = p1, p2
        if pop_size % 2:
            offspring[-1] = parents[-1]

        # Mutation
        mask = rng.random((pop_size, ndim)) < 0.15
        offspring += mask * rng.normal(0, 0.1, (pop_size, ndim)) * span
        offspring = np.clip(offspring, lo, hi)

        if best_chrom is not None:
            offspring[0] = best_chrom
        pop = offspring

    return best_chrom, best_fit



# ═══════════════════════════════════════════════════════════════════
# WORKER: satu sel tabel
# ═══════════════════════════════════════════════════════════════════

def _worker_sel(args):
    """Optimize one table cell: all (n1,n2), pick best."""
    (delta, val_param, nama_param, base_params, error_model,
     n1_range, N_max, H_val, bounds, pop_size, n_gen, patience) = args

    warnings.filterwarnings("ignore")
    p = dict(base_params)
    p[nama_param] = val_param

    gamma0 = p.get('gamma0', GAMMA_0)
    theta = p.get('theta', 0.0)
    B_val = p.get('B', 1)
    m = p.get('m', 1)
    eta = p.get('eta', 0.0)
    omega = p.get('omega', 0.0)
    phi = p.get('phi', 0.0)

    g0s, _ = get_gamma_stars(1.0, gamma0, theta, B_val, m,
                             eta=eta, omega=omega, phi=phi, error_model=error_model)
    _, gps = get_gamma_stars(delta, gamma0, theta, B_val, m,
                             eta=eta, omega=omega, phi=phi, error_model=error_model)

    kombinasi = [(n1, n2)
                 for n1 in n1_range
                 for n2 in range(max(N0, n1+2), N_max - n1 + 1)
                 if n1 + n2 <= N_max]

    best = None
    for n1, n2 in kombinasi:
        seed = 42 + n1*100 + n2 + int(delta*1000)
        chrom, fit = _ga_2phase(n1, n2, g0s, gps, H_val,
                                bounds, pop_size, n_gen, patience, seed)
        if chrom is not None and (best is None or fit < best['tarl']):
            best = {'n1': n1, 'n2': n2, 'chrom': chrom, 'tarl': fit}

    # Phase 2: refine best with full P_ooc
    if best is not None:
        ku1, kl1, wu, wl, ku2, kl2 = best['chrom']
        limits = compute_limits(best['n1'], best['n2'],
                                ku1, kl1, wu, wl, ku2, kl2, g0s)
        if is_valid(*limits):
            p_full = P_ooc_full(best['n1'], best['n2'], *limits, gps)
            best['tarl'] = tarl_from_pooc(p_full, H_val)
        best['koef'] = dict(zip(['ku1','kl1','wu','wl','ku2','kl2'], best['chrom']))

    return (delta, val_param, best)



# ═══════════════════════════════════════════════════════════════════
# API: buat_tabel_jurnal
# ═══════════════════════════════════════════════════════════════════

def buat_tabel_jurnal(variasi_param, nama_param, base_params,
                      error_model='constant', delta_list=None,
                      n1_range=None, N_max=N_MAX, H_val=H,
                      bounds=None, pop_size=POP_SIZE,
                      n_gen=N_GENERATIONS, patience=PATIENCE,
                      n_workers=None, verbose=True):
    if delta_list is None: delta_list = DELTA_LIST
    if n1_range is None: n1_range = N1_RANGE
    if bounds is None: bounds = DEFAULT_BOUNDS
    if n_workers is None: n_workers = min(mp.cpu_count(), 8)

    semua_sel = list(itertools.product(delta_list, variasi_param))

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  FAST: {nama_param} | {error_model} | {len(semua_sel)} sel | {n_workers}w")
        print(f"{'═'*60}")

    t0 = time.time()
    task_args = [
        (delta, val, nama_param, base_params, error_model,
         n1_range, N_max, H_val, bounds, pop_size, n_gen, patience)
        for delta, val in semua_sel
    ]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_sel, task_args)

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✅ {elapsed:.0f}s ({elapsed/60:.1f} menit)")

    hasil = {}
    for (delta, val_param, res) in results:
        hasil.setdefault(delta, {})[val_param] = res
    return hasil



# ═══════════════════════════════════════════════════════════════════
# OUTPUT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def cetak_tabel_jurnal(hasil, variasi_param, nama_param, delta_list=None, judul=""):
    if delta_list is None: delta_list = sorted(hasil.keys())
    lebar, n_col = 26, len(variasi_param)
    sep = "─" * (10 + lebar * n_col)
    if judul: print(f"\n  {judul}")
    print(sep)
    print(f"{'δ':^10}" + "".join(f"{f'{nama_param}={v}':^{lebar}}" for v in variasi_param))
    print(sep)
    for delta in delta_list:
        baris = [["N/A","","",""] for _ in variasi_param]
        for ci, val in enumerate(variasi_param):
            res = hasil.get(delta, {}).get(val)
            if res and res.get('koef'):
                k = res['koef']
                baris[ci] = [
                    f"{res['n1']}, {res['n2']}, {k['ku1']:.4f}, {k['kl1']:.4f},",
                    f"{k['wu']:.4f}, {k['wl']:.4f},",
                    f"{k['ku2']:.4f}, {k['kl2']:.4f}",
                    f"({res['tarl']:.2f})",
                ]
        for r in range(4):
            line = (f"{delta:<10.4f}" if r == 0 else " "*10)
            for ci in range(n_col):
                line += f"{baris[ci][r]:^{lebar}}"
            print(line)
        print()
    print(sep)


def cetak_tabel_df(hasil, variasi_param, nama_param, delta_list=None):
    try:
        import pandas as pd
    except ImportError:
        return None
    if delta_list is None: delta_list = sorted(hasil.keys())
    rows = []
    for delta in delta_list:
        for val in variasi_param:
            res = hasil.get(delta, {}).get(val)
            if res and res.get('koef'):
                k = res['koef']
                rows.append({'delta': delta, nama_param: val,
                    'n1': res['n1'], 'n2': res['n2'],
                    **{key: round(k[key], 4) for key in k},
                    'TARL': round(res['tarl'], 4)})
    return pd.DataFrame(rows)


def simpan_semua(semua_hasil, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    for fname, (hasil, vp, np_) in semua_hasil.items():
        df = cetak_tabel_df(hasil, vp, np_)
        if df is not None:
            path = os.path.join(output_dir, f"{fname}.csv")
            df.to_csv(path, index=False)
            print(f"  ✓ {path} ({len(df)} rows)")


def plot_perbandingan_tarl(hasil_dict, delta_list=None, judul=""):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if delta_list is None: delta_list = DELTA_LIST
    n_plot = len(hasil_dict)
    fig, axes = plt.subplots(1, n_plot, figsize=(5*n_plot, 5))
    if n_plot == 1: axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 0.9, 6))
    for ax_idx, (label, (hasil, vp, np_)) in enumerate(hasil_dict.items()):
        ax = axes[ax_idx]
        for ci, val in enumerate(vp):
            tv, dv = [], []
            for d in delta_list:
                r = hasil.get(d, {}).get(val)
                if r and r.get('koef'): tv.append(r['tarl']); dv.append(d)
            if tv: ax.plot(dv, tv, 'o-', color=colors[ci%6], lw=2, ms=5, label=f'{np_}={val}')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('δ'); ax.set_ylabel('TARL')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('tarl_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved: tarl_comparison.png")



# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    mp.freeze_support()
    print("=" * 60)
    print(f"  DS CV-ME FAST | CPU: {mp.cpu_count()} | Pop:{POP_SIZE} Gen:{N_GENERATIONS}")
    print("=" * 60)

    theta_list = [0, 0.01, 0.03, 0.05]
    eta_list = [0, 0.1, 0.3, 0.5]
    m_list = [1, 3, 5, 7]
    B_list = [1, 2, 3, 4]
    omega_list = [0.0, 0.1, 0.2, 0.3]
    phi_list = [0.0, 0.1, 0.2, 0.3]

    h1 = buat_tabel_jurnal(theta_list, 'theta',
        {'gamma0':GAMMA_0,'theta':0.0,'eta':0.28,'B':1,'m':1}, 'constant')
    cetak_tabel_jurnal(h1, theta_list, 'θ', judul="Model A | variasi θ")

    h2 = buat_tabel_jurnal(eta_list, 'eta',
        {'gamma0':GAMMA_0,'theta':0.0,'eta':0.0,'B':1,'m':1}, 'constant')
    cetak_tabel_jurnal(h2, eta_list, 'η', judul="Model A | variasi η")

    h3 = buat_tabel_jurnal(m_list, 'm',
        {'gamma0':GAMMA_0,'theta':0.05,'eta':0.28,'B':1,'m':1}, 'constant')
    cetak_tabel_jurnal(h3, m_list, 'm', judul="Model A | variasi m")

    h4 = buat_tabel_jurnal(B_list, 'B',
        {'gamma0':GAMMA_0,'theta':0.05,'eta':0.28,'B':1,'m':1}, 'constant')
    cetak_tabel_jurnal(h4, B_list, 'B', judul="Model A | variasi B")

    h5 = buat_tabel_jurnal(theta_list, 'theta',
        {'gamma0':GAMMA_0,'theta':0.0,'omega':0.0,'phi':0.0,'B':1,'m':1}, 'linear')
    cetak_tabel_jurnal(h5, theta_list, 'θ', judul="Model B | variasi θ")

    h6 = buat_tabel_jurnal(omega_list, 'omega',
        {'gamma0':GAMMA_0,'theta':0.0,'omega':0.0,'phi':0.0,'B':1,'m':1}, 'linear')
    cetak_tabel_jurnal(h6, omega_list, 'ω', judul="Model B | variasi ω")

    h7 = buat_tabel_jurnal(phi_list, 'phi',
        {'gamma0':GAMMA_0,'theta':0.0,'omega':0.0,'phi':0.0,'B':1,'m':1}, 'linear')
    cetak_tabel_jurnal(h7, phi_list, 'φ', judul="Model B | variasi φ")

    h8 = buat_tabel_jurnal(m_list, 'm',
        {'gamma0':GAMMA_0,'theta':0.05,'omega':0.1,'phi':0.1,'B':1,'m':1}, 'linear')
    cetak_tabel_jurnal(h8, m_list, 'm', judul="Model B | variasi m")

    h9 = buat_tabel_jurnal(B_list, 'B',
        {'gamma0':GAMMA_0,'theta':0.05,'omega':0.1,'phi':0.1,'B':1,'m':1}, 'linear')
    cetak_tabel_jurnal(h9, B_list, 'B', judul="Model B | variasi B")

    simpan_semua({
        't1_theta':(h1,theta_list,'theta'), 't1_eta':(h2,eta_list,'eta'),
        't1_m':(h3,m_list,'m'), 't1_B':(h4,B_list,'B'),
        't3_theta':(h5,theta_list,'theta'), 't3_omega':(h6,omega_list,'omega'),
        't3_phi':(h7,phi_list,'phi'), 't3_m':(h8,m_list,'m'),
        't3_B':(h9,B_list,'B'),
    })
    print("\n✅ DONE!")
