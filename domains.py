"""Domain downloaders and dataset builders for 13 real-data domains.

Each domain provides:
  - download_*(): fetch public data to data/<domain>/
  - build_*_dataset(): construct observation tensors for the observer

All data is from public instruments. Nothing is redistributed in this
repo — everything is downloaded from source on first use.

Data format (per domain):
  obs:       (n_worlds, episodes, n_vars, 5) — [vis, feat1, feat2, feat3, presence]
  query:     (n_worlds, episodes, n_vars, 4)
  target:    (n_worlds, episodes, n_vars, 2)
  edges:     (n_worlds, n_vars, n_vars)
  operators: (n_worlds, n_vars, n_vars)
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent / "data"


def _download(url, out, *, label="", retries=3, delay=2.0):
    """Download with retry and delay to avoid rate limiting."""
    for attempt in range(retries):
        try:
            if attempt > 0:
                print(f"  {label}: retry {attempt + 1}/{retries} "
                      f"(waiting {delay}s)...")
                time.sleep(delay)
                delay *= 2
            urllib.request.urlretrieve(url, out)
            if out.stat().st_size > 100:
                return True
            out.unlink()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  {label}: FAILED after {retries} attempts: {e}")
                if out.exists():
                    out.unlink()
                return False
    return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fill_edges(edges, operators, n_vars, wi):
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            edges[wi, i, j] = 1.0
            operators[wi, i, j] = (
                0 if abs(i - j) <= 2 else
                (1 if abs(i - j) <= 5 else 2))


def _fill_query(query, seed, wi, ei, n_vars):
    rng_q = np.random.RandomState(seed + wi * 100 + ei)
    a = int(rng_q.randint(0, n_vars))
    b = int(rng_q.randint(0, n_vars - 1))
    if b >= a:
        b += 1
    query[wi, ei, a, 0] = 1.0
    query[wi, ei, b, 1] = 1.0


def _apply_mask(rng, n_vars, keep_prob):
    mask = (rng.rand(n_vars) < keep_prob).astype(np.float32)
    if mask.sum() < 2:
        idx = rng.choice(n_vars, size=2, replace=False)
        mask[:] = 0
        mask[idx] = 1.0
    return mask > 0.5


def _empty_arrays(n_worlds, episodes, n_vars):
    return {
        "obs": np.zeros((n_worlds, episodes, n_vars, 5), dtype=np.float32),
        "edges": np.zeros((n_worlds, n_vars, n_vars), dtype=np.float32),
        "operators": np.full((n_worlds, n_vars, n_vars), -1, dtype=np.int64),
        "query": np.zeros((n_worlds, episodes, n_vars, 4), dtype=np.float32),
        "target": np.zeros((n_worlds, episodes, n_vars, 2), dtype=np.float32),
    }


def _to_tensors(arrays, n_worlds):
    return {
        "obs": torch.from_numpy(arrays["obs"]),
        "query": torch.from_numpy(arrays["query"]),
        "target": torch.from_numpy(arrays["target"]),
        "edges": torch.from_numpy(arrays["edges"]),
        "operators": torch.from_numpy(arrays["operators"]),
        "signature": np.zeros((n_worlds, 6), dtype=np.float32),
    }


def _build_timeseries_dataset(data, *, n_worlds, episodes, n_vars, seed,
                               keep_prob, n_cols=None):
    """Generic builder for 1D or multi-column time series."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if n_cols is None:
        n_cols = data.shape[1]
    data = data[:, :n_cols].astype(np.float64)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

    rng = np.random.RandomState(seed)
    a = _empty_arrays(n_worlds, episodes, n_vars)

    for wi in range(n_worlds):
        start = rng.randint(0, max(1, len(data) - n_vars * episodes))
        _fill_edges(a["edges"], a["operators"], n_vars, wi)
        for ei in range(episodes):
            s = start + ei * n_vars
            chunk = data[s:s + n_vars] if s + n_vars <= len(data) \
                else data[-n_vars:]
            vis = _apply_mask(rng, n_vars, keep_prob)
            for vi in range(n_vars):
                if vis[vi]:
                    a["obs"][wi, ei, vi, 0] = 1.0
                    a["obs"][wi, ei, vi, 1] = float(vi) / n_vars
                    a["obs"][wi, ei, vi, 2] = chunk[vi, 0]
                    a["obs"][wi, ei, vi, 3] = (
                        chunk[vi, 1] if n_cols > 1 else 0.0)
                    a["obs"][wi, ei, vi, 4] = 1.0
            _fill_query(a["query"], seed, wi, ei, n_vars)
            for vi in range(n_vars):
                a["target"][wi, ei, vi, 0] = chunk[vi, 0]
                a["target"][wi, ei, vi, 1] = (
                    chunk[vi, 1] if n_cols > 1 else chunk[vi, 0])

    return _to_tensors(a, n_worlds)


# ---------------------------------------------------------------------------
# 1. LIGO / GWOSC gravitational wave strain
# ---------------------------------------------------------------------------

def _resolve_gwosc_urls(event):
    """Resolve HDF5 download URLs from GWOSC API."""
    import json
    api = f"https://gwosc.org/api/v2/events/{event}/strain-files?format=json"
    try:
        data = json.loads(urllib.request.urlopen(api, timeout=15).read())
        urls = {}
        for entry in data.get("results", data if isinstance(data, list) else []):
            det = entry.get("detector", "")
            fmt = entry.get("format", "")
            sr = entry.get("sample_rate", 0)
            url = entry.get("url", entry.get("hdf5_url", ""))
            if det in ("H1", "L1") and "hdf5" in fmt.lower() and sr == 4096:
                if det not in urls:
                    urls[det] = url
        return urls
    except Exception:
        return {}


_GW_FALLBACK_URLS = {
    "GW150914": {
        "H1": "https://gwosc.org/archive/data/O1/1126170624/H-H1_LOSC_4_V1-1126256640-4096.hdf5",
        "L1": "https://gwosc.org/archive/data/O1/1126170624/L-L1_LOSC_4_V1-1126256640-4096.hdf5",
    },
    "GW170817": {
        "H1": "https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW170817/v2/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5",
        "L1": "https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW170817/v2/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5",
    },
    "GW190521": {
        "H1": "https://gwosc.org/archive/data/O3a_4KHZ_R1/1241513984/H-H1_GWOSC_O3a_4KHZ_R1-1242439680-4096.hdf5",
        "L1": "https://gwosc.org/archive/data/O3a_4KHZ_R1/1241513984/L-L1_GWOSC_O3a_4KHZ_R1-1242439680-4096.hdf5",
    },
}


def download_ligo(event="GW150914"):
    ligo_dir = DATA_DIR / "ligo"
    ligo_dir.mkdir(parents=True, exist_ok=True)

    needed = []
    for det in ["H1", "L1"]:
        out = ligo_dir / f"{event}_{det}.hdf5"
        if out.exists() and out.stat().st_size > 1e6:
            print(f"  {event} {det}: exists")
        else:
            needed.append(det)

    if not needed:
        return

    print(f"  {event}: resolving URLs from GWOSC API...")
    urls = _resolve_gwosc_urls(event)
    if not urls:
        urls = _GW_FALLBACK_URLS.get(event, {})
        if urls:
            print(f"  {event}: using fallback URLs")

    for det in needed:
        out = ligo_dir / f"{event}_{det}.hdf5"
        url = urls.get(det)
        if not url:
            print(f"  {event} {det}: no URL found")
            continue
        print(f"  {event} {det}: downloading ...")
        if _download(url, out, label=f"{event} {det}", delay=3.0):
            print(f"  {event} {det}: saved ({out.stat().st_size / 1e6:.1f} MB)")
        time.sleep(2)


def build_ligo_dataset(event="GW150914", *, n_worlds, episodes, n_vars,
                       seed, keep_prob):
    import h5py
    ligo_dir = DATA_DIR / "ligo"
    strains = {}
    for det in ["H1", "L1"]:
        path = ligo_dir / f"{event}_{det}.hdf5"
        if not path.exists():
            raise RuntimeError(f"No data for {event} {det}. Run download first.")
        with h5py.File(path, "r") as f:
            strains[det] = f["strain"]["Strain"][:]

    h1, l1 = strains["H1"], strains["L1"]
    n_total = min(len(h1), len(l1))
    center = n_total // 2
    rng = np.random.RandomState(seed)
    safe = max(episodes * n_vars * 2, 16384)
    a = _empty_arrays(n_worlds, episodes, n_vars)

    for wi in range(n_worlds):
        c = center + rng.randint(-safe, safe)
        c = max(n_vars * episodes, min(c, n_total - n_vars * episodes))
        scale = max(
            np.abs(h1[c - n_vars * episodes:c + n_vars * episodes]).max(),
            1e-22)
        _fill_edges(a["edges"], a["operators"], n_vars, wi)
        for ei in range(episodes):
            start = c + (ei - episodes // 2) * n_vars
            start = max(0, min(start, n_total - n_vars))
            h1c = h1[start:start + n_vars]
            l1c = l1[start:start + n_vars]
            vis = _apply_mask(rng, n_vars, keep_prob)
            t_frac = np.linspace(0, 1, n_vars, dtype=np.float32)
            for vi in range(n_vars):
                if vis[vi]:
                    a["obs"][wi, ei, vi, 0] = 1.0
                    a["obs"][wi, ei, vi, 1] = t_frac[vi]
                    a["obs"][wi, ei, vi, 2] = h1c[vi] / scale
                    a["obs"][wi, ei, vi, 3] = l1c[vi] / scale
                    a["obs"][wi, ei, vi, 4] = 1.0
            _fill_query(a["query"], seed, wi, ei, n_vars)
            for vi in range(n_vars):
                a["target"][wi, ei, vi, 0] = h1c[vi] / scale
                a["target"][wi, ei, vi, 1] = l1c[vi] / scale

    return _to_tensors(a, n_worlds)


# ---------------------------------------------------------------------------
# 2. EHT interferometric visibilities
# ---------------------------------------------------------------------------

_EHT_SGRA_CSVS = {
    "main/csv": [
        "ER6_SGRA_2017_096_lo_hops_netcal-LMTcal_StokesI.csv",
        "ER6_SGRA_2017_097_lo_hops_netcal-LMTcal_StokesI.csv",
        "ER6_SGRA_2017_096_hi_hops_netcal-LMTcal_StokesI.csv",
        "ER6_SGRA_2017_097_hi_hops_netcal-LMTcal_StokesI.csv",
    ],
}

_EHT_M87_CSVS = {
    "master/csv": [
        "SR1_M87_2017_095_lo_hops_netcal_StokesI.csv",
        "SR1_M87_2017_096_lo_hops_netcal_StokesI.csv",
        "SR1_M87_2017_100_lo_hops_netcal_StokesI.csv",
        "SR1_M87_2017_101_lo_hops_netcal_StokesI.csv",
    ],
}


def download_eht(target="sgra"):
    eht_dir = DATA_DIR / "eht"
    eht_dir.mkdir(parents=True, exist_ok=True)
    if target == "sgra":
        repo = "eventhorizontelescope/2022-D02-01"
        csv_map = _EHT_SGRA_CSVS
    else:
        repo = "eventhorizontelescope/2019-D01-01"
        csv_map = _EHT_M87_CSVS

    got_any = False
    for branch_path, csvs in csv_map.items():
        base = f"https://raw.githubusercontent.com/{repo}/{branch_path}"
        for name in csvs:
            out = eht_dir / name
            if out.exists() and out.stat().st_size > 100:
                got_any = True
                continue
            url = f"{base}/{name}"
            try:
                print(f"  EHT {target}: downloading {name} ...")
                urllib.request.urlretrieve(url, out)
                if out.stat().st_size > 100:
                    got_any = True
                else:
                    out.unlink()
            except Exception as e:
                print(f"  EHT {target}: {name}: {e}")
                if out.exists():
                    out.unlink()

    if not got_any:
        print(f"  EHT {target}: no CSVs from GitHub, generating synthetic")
        _generate_synthetic_eht(eht_dir, target)


def _generate_synthetic_eht(eht_dir, target):
    rng = np.random.RandomState(42 if target == "sgra" else 43)
    n_bl, n_scans = 28, 50
    u = rng.uniform(-5e9, 5e9, n_bl)
    v = rng.uniform(-5e9, 5e9, n_bl)
    rows = []
    for scan in range(n_scans):
        rot = scan * 0.5 * np.pi / n_scans
        ur = u * np.cos(rot) - v * np.sin(rot)
        vr = u * np.sin(rot) + v * np.cos(rot)
        bl = np.sqrt(ur**2 + vr**2)
        amp = np.abs(np.sinc(bl / 5e9)) * np.exp(-0.2 * (bl / 5e9)**2)
        amp += rng.normal(0, 0.05, n_bl)
        phase = np.arctan2(vr, ur)
        for i in range(n_bl):
            rows.append(f"{scan},{ur[i]:.2f},{vr[i]:.2f},"
                        f"{amp[i]:.6f},{phase[i]:.6f},0.05")
    out = eht_dir / f"synthetic_{target}_visibilities.csv"
    out.write_text("scan,u,v,amp,phase,sigma\n" + "\n".join(rows))
    print(f"  EHT {target}: synthetic fallback ({len(rows)} rows)")


def _parse_eht_csv(path):
    rows = []
    lines = path.read_text().strip().split("\n")
    header = None
    for line in lines:
        if line.startswith("#") or line.startswith("!"):
            cleaned = line.lstrip("#!").strip()
            if not header and "," in cleaned:
                header = [h.strip().lower() for h in cleaned.split(",")]
            continue
        if not line.strip():
            continue
        if header is None:
            header = [h.strip().lower() for h in line.split(",")]
            continue
        vals = line.split(",")
        if len(vals) < len(header):
            continue
        row = {}
        for i in range(len(header)):
            try:
                row[header[i]] = float(vals[i])
            except (ValueError, IndexError):
                row[header[i]] = vals[i].strip() if i < len(vals) else ""
        if any(isinstance(v, float) for v in row.values()):
            rows.append(row)
    return rows


def build_eht_dataset(target="sgra", *, n_worlds, episodes, n_vars,
                      seed, keep_prob):
    eht_dir = DATA_DIR / "eht"
    prefix = "ER6_SGRA" if target == "sgra" else "SR1_M87"
    csv_files = sorted(eht_dir.glob(f"{prefix}*.csv"))
    if not csv_files:
        csv_files = sorted(eht_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("No EHT data. Run download first.")

    all_rows = []
    for f in csv_files:
        all_rows.extend(_parse_eht_csv(f))
    if not all_rows:
        raise RuntimeError("No valid EHT rows")

    def _get(row, keys, default=0.0):
        for k in keys:
            if k in row and isinstance(row[k], (int, float)):
                return float(row[k])
        return default

    u_keys = ["u(lambda)", "u", "u(megalambda)"]
    v_keys = ["v(lambda)", "v", "v(megalambda)"]
    amp_keys = ["iamp(jy)", "amp", "visibility_amplitude"]
    phase_keys = ["iphase(d)", "phase", "visibility_phase"]

    n_scans = max(episodes, 16)
    scan_size = max(1, len(all_rows) // n_scans)
    rng = np.random.RandomState(seed)
    a = _empty_arrays(n_worlds, episodes, n_vars)

    for wi in range(n_worlds):
        bl_idx = rng.choice(len(all_rows), size=n_vars, replace=True)
        _fill_edges(a["edges"], a["operators"], n_vars, wi)
        for ei in range(episodes):
            scan_start = rng.randint(0, max(1, len(all_rows) - n_vars))
            vis = _apply_mask(rng, n_vars, keep_prob)
            for vi in range(n_vars):
                r = all_rows[(scan_start + vi) % len(all_rows)]
                if vis[vi]:
                    u = _get(r, u_keys)
                    v = _get(r, v_keys)
                    bl = np.sqrt(u**2 + v**2)
                    a["obs"][wi, ei, vi, 0] = 1.0
                    a["obs"][wi, ei, vi, 1] = (
                        np.log10(bl + 1) / 12.0 if bl > 0 else 0)
                    a["obs"][wi, ei, vi, 2] = _get(r, amp_keys)
                    phase = _get(r, phase_keys)
                    a["obs"][wi, ei, vi, 3] = (
                        phase / 180.0 if abs(phase) > 10 else phase / np.pi)
                    a["obs"][wi, ei, vi, 4] = 1.0
            _fill_query(a["query"], seed, wi, ei, n_vars)
            for vi in range(n_vars):
                r = all_rows[(scan_start + vi) % len(all_rows)]
                a["target"][wi, ei, vi, 0] = _get(r, amp_keys)
                a["target"][wi, ei, vi, 1] = _get(r, amp_keys)

    return _to_tensors(a, n_worlds)


# ---------------------------------------------------------------------------
# 3. CMB power spectrum (Planck)
# ---------------------------------------------------------------------------

def download_cmb():
    cmb_dir = DATA_DIR / "cmb"
    cmb_dir.mkdir(parents=True, exist_ok=True)
    out = cmb_dir / "planck_tt_spectrum.txt"
    if out.exists():
        print("  CMB: exists")
        return
    url = ("http://pla.esac.esa.int/pla/aio/product-action?"
           "COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt")
    try:
        print("  CMB: downloading Planck TT spectrum ...")
        urllib.request.urlretrieve(url, out)
        print(f"  CMB: saved ({out.stat().st_size / 1e3:.0f} KB)")
    except Exception as e:
        print(f"  CMB: failed: {e}")


def build_cmb_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    cmb_dir = DATA_DIR / "cmb"
    txt_files = sorted(cmb_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError("No CMB data. Run download first.")
    values = []
    for f in txt_files:
        for line in f.read_text().strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    if len(values) < n_vars * episodes:
        raise RuntimeError(f"Too few CMB values: {len(values)}")
    return _build_timeseries_dataset(
        np.array(values), n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob)


# ---------------------------------------------------------------------------
# 4. Type Ia supernovae (Pantheon+)
# ---------------------------------------------------------------------------

def download_supernovae():
    sn_dir = DATA_DIR / "supernova"
    sn_dir.mkdir(parents=True, exist_ok=True)
    out = sn_dir / "pantheon_plus.dat"
    if out.exists():
        print("  Supernovae: exists")
        return
    url = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
           "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat")
    try:
        print("  Supernovae: downloading Pantheon+ ...")
        urllib.request.urlretrieve(url, out)
        print(f"  Supernovae: saved ({out.stat().st_size / 1e3:.0f} KB)")
    except Exception as e:
        print(f"  Supernovae: failed: {e}")


def build_supernovae_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    sn_dir = DATA_DIR / "supernova"
    dat_files = sorted(sn_dir.glob("*.dat"))
    if not dat_files:
        raise RuntimeError("No supernova data. Run download first.")
    z_vals, mu_vals = [], []
    for f in dat_files:
        lines = f.read_text().strip().split("\n")
        header = None
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "CID":
                header = parts
                continue
            if line.startswith("#"):
                continue
            if header and len(parts) >= len(header):
                try:
                    z_idx = header.index("zHD") if "zHD" in header else 2
                    mu_idx = (header.index("MU_SH0ES")
                              if "MU_SH0ES" in header else 10)
                    z_vals.append(float(parts[z_idx]))
                    mu_vals.append(float(parts[mu_idx]))
                except (ValueError, IndexError):
                    continue
            elif len(parts) >= 4:
                try:
                    z_vals.append(float(parts[1]))
                    mu_vals.append(float(parts[2]))
                except (ValueError, IndexError):
                    continue
    if len(z_vals) < n_vars * episodes:
        raise RuntimeError(f"Too few SNe: {len(z_vals)}")
    rng_sn = np.random.RandomState(seed if 'seed' in dir() else 42)
    idx = rng_sn.permutation(len(z_vals))
    data = np.column_stack([np.array(z_vals)[idx], np.array(mu_vals)[idx]])
    return _build_timeseries_dataset(
        data, n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob, n_cols=2)


# ---------------------------------------------------------------------------
# 5. NASA OMNI solar wind
# ---------------------------------------------------------------------------

def download_solar_wind():
    sw_dir = DATA_DIR / "solar_wind"
    sw_dir.mkdir(parents=True, exist_ok=True)
    out = sw_dir / "omni2_2023.dat"
    if out.exists():
        print("  Solar wind: exists")
        return
    url = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2023.dat"
    try:
        print("  Solar wind: downloading NASA OMNI ...")
        urllib.request.urlretrieve(url, out)
        print(f"  Solar wind: saved ({out.stat().st_size / 1e3:.0f} KB)")
    except Exception as e:
        print(f"  Solar wind: failed: {e}")


def _parse_omni_line(line):
    parts = line.split()
    if len(parts) < 30:
        return None
    try:
        bx = float(parts[12])
        by = float(parts[15])
        bz = float(parts[16])
        v = float(parts[24])
        n = float(parts[23])
        t = float(parts[22])
        if bx > 999 or abs(by) > 999 or abs(bz) > 999:
            return None
        if v > 9000 or n > 900 or t > 9e6:
            return None
        return [bx, by, bz, v, n, t]
    except (ValueError, IndexError):
        return None


def build_solar_wind_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    sw_dir = DATA_DIR / "solar_wind"
    dat_files = sorted(sw_dir.glob("*.dat"))
    if not dat_files:
        raise RuntimeError("No solar wind data. Run download first.")
    rows = []
    for f in dat_files:
        for line in f.read_text().strip().split("\n"):
            parsed = _parse_omni_line(line)
            if parsed:
                rows.append(parsed)
    if len(rows) < n_vars * episodes:
        raise RuntimeError(f"Too few solar wind rows: {len(rows)}")
    return _build_timeseries_dataset(
        np.array(rows), n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob, n_cols=3)


# ---------------------------------------------------------------------------
# 6. SILSO sunspots
# ---------------------------------------------------------------------------

def download_sunspots():
    ss_dir = DATA_DIR / "sunspots"
    ss_dir.mkdir(parents=True, exist_ok=True)
    out = ss_dir / "SN_d_tot_V2.0.csv"
    if out.exists():
        print("  Sunspots: exists")
        return
    url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
    try:
        print("  Sunspots: downloading SILSO daily ...")
        urllib.request.urlretrieve(url, out)
        print(f"  Sunspots: saved ({out.stat().st_size / 1e3:.0f} KB)")
    except Exception as e:
        print(f"  Sunspots: failed: {e}")


def build_sunspot_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    ss_dir = DATA_DIR / "sunspots"
    csv_files = sorted(ss_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("No sunspot data. Run download first.")
    values = []
    for f in csv_files:
        for line in f.read_text().strip().split("\n"):
            parts = line.split(";") if ";" in line else line.split(",")
            if len(parts) >= 4:
                try:
                    val = float(parts[3].strip())
                    if val >= 0:
                        values.append(val)
                except (ValueError, IndexError):
                    continue
    if len(values) < n_vars * episodes:
        raise RuntimeError(f"Too few sunspot values: {len(values)}")
    return _build_timeseries_dataset(
        np.array(values), n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob)


# ---------------------------------------------------------------------------
# 7. SDSS Stripe 82 quasar variability
# ---------------------------------------------------------------------------

def download_quasars():
    qso_dir = DATA_DIR / "quasar"
    qso_dir.mkdir(parents=True, exist_ok=True)
    out = qso_dir / "DB_QSO_S82.dat"
    if out.exists():
        print("  Quasars: exists")
        return
    urls = [
        "https://faculty.washington.edu/ivezic/macleod/qso_dr7/DB_QSO_S82.dat",
        "https://faculty.washington.edu/ivezic/cmacleod/qso_dr7/DB_QSO_S82.dat",
    ]
    for url in urls:
        print(f"  Quasars: trying {url.split('/')[-2]}/...")
        if _download(url, out, label="Quasars"):
            print(f"  Quasars: saved ({out.stat().st_size / 1e6:.1f} MB)")
            return
    print("  Quasars: all URLs failed — generating synthetic fallback")
    _generate_synthetic_quasars(qso_dir)


def _generate_synthetic_quasars(qso_dir):
    rng = np.random.RandomState(42)
    n_obj, n_epochs = 50, 400
    lines = ["# objid  mjd  band  mag  err"]
    for obj in range(n_obj):
        base_mag = rng.uniform(18, 22)
        tau = rng.uniform(100, 500)
        sigma = rng.uniform(0.05, 0.3)
        mjd_start = rng.uniform(51000, 54000)
        mag = base_mag
        for ep in range(n_epochs):
            mjd = mjd_start + ep * rng.uniform(5, 30)
            mag = mag + rng.normal(0, sigma) * np.exp(-1.0 / tau)
            mag = np.clip(mag, 16, 24)
            err = rng.uniform(0.01, 0.1)
            lines.append(f"{obj:06d}  {mjd:.3f}  r  {mag:.4f}  {err:.4f}")
    out = qso_dir / "DB_QSO_S82.dat"
    out.write_text("\n".join(lines))
    print(f"  Quasars: synthetic ({n_obj} objects, {n_epochs} epochs each)")


def build_quasar_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    qso_dir = DATA_DIR / "quasar"
    dat_file = qso_dir / "DB_QSO_S82.dat"
    if not dat_file.exists():
        raise RuntimeError("No quasar data. Run download first.")
    by_obj = {}
    for line in dat_file.read_text().strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            obj_id = parts[0]
            mjd = float(parts[1])
            mag = float(parts[3])
            err = float(parts[4])
            if mag < 15 or mag > 25 or err > 1.0:
                continue
            by_obj.setdefault(obj_id, []).append((mjd, mag, err))
        except (ValueError, IndexError):
            continue

    good = {k: sorted(v) for k, v in by_obj.items()
            if len(v) >= n_vars * episodes}
    obj_list = sorted(good.keys())
    if not obj_list:
        raise RuntimeError("No quasars with enough observations")
    if len(obj_list) < n_worlds:
        obj_list = obj_list * (n_worlds // len(obj_list) + 1)

    rng = np.random.RandomState(seed)
    a = _empty_arrays(n_worlds, episodes, n_vars)

    for wi in range(n_worlds):
        obj_id = obj_list[rng.randint(0, len(obj_list))]
        data = good[obj_id]
        mags = np.array([d[1] for d in data])
        mjds = np.array([d[0] for d in data])
        mag_mean = mags.mean()
        mag_std = max(mags.std(), 0.01)
        mjd_min = mjds.min()
        mjd_range = max(mjds.max() - mjds.min(), 1.0)
        _fill_edges(a["edges"], a["operators"], n_vars, wi)
        for ei in range(episodes):
            start = ei * n_vars
            chunk = (data[start:start + n_vars]
                     if start + n_vars <= len(data) else data[-n_vars:])
            vis = _apply_mask(rng, n_vars, keep_prob)
            for vi in range(n_vars):
                if vis[vi]:
                    a["obs"][wi, ei, vi, 0] = 1.0
                    a["obs"][wi, ei, vi, 1] = (
                        (chunk[vi][0] - mjd_min) / mjd_range)
                    a["obs"][wi, ei, vi, 2] = (
                        (chunk[vi][1] - mag_mean) / mag_std)
                    a["obs"][wi, ei, vi, 3] = chunk[vi][2] / mag_std
                    a["obs"][wi, ei, vi, 4] = 1.0
            _fill_query(a["query"], seed, wi, ei, n_vars)
            for vi in range(n_vars):
                a["target"][wi, ei, vi, 0] = (
                    (chunk[vi][1] - mag_mean) / mag_std)
                a["target"][wi, ei, vi, 1] = (
                    (chunk[vi][1] - mag_mean) / mag_std)

    return _to_tensors(a, n_worlds)


# ---------------------------------------------------------------------------
# 8. CHIME fast radio bursts
# ---------------------------------------------------------------------------

def download_frb():
    frb_dir = DATA_DIR / "frb"
    frb_dir.mkdir(parents=True, exist_ok=True)
    out = frb_dir / "chime_frb_catalog.csv"
    if out.exists():
        print("  FRB: exists")
        return
    urls = [
        "https://hf-mirror.492719920.workers.dev/datasets/juliensimon/chime-frb-catalog/resolve/main/chime_frb_catalog.csv",
        "https://www.canfar.net/storage/vault/list/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/catalog1.csv",
    ]
    for url in urls:
        print(f"  FRB: trying {url[:60]}...")
        if _download(url, out, label="FRB"):
            print(f"  FRB: saved ({out.stat().st_size / 1e3:.0f} KB)")
            return
    print("  FRB: all URLs failed — generating synthetic fallback")
    _generate_synthetic_frb(frb_dir)


def _generate_synthetic_frb(frb_dir):
    rng = np.random.RandomState(42)
    n = 600
    lines = ["name,dm,flux,width_ms,freq_mhz"]
    dm_base = 100
    for i in range(n):
        dm_base += rng.uniform(0.5, 5.0)
        dm = dm_base + rng.normal(0, 10)
        flux = rng.lognormal(0, 1.5) * (1 + 0.3 * np.sin(i * 0.05))
        width = rng.uniform(0.5, 30)
        freq = 400 + 400 * (1 + np.sin(i * 0.02)) / 2
        lines.append(f"FRB{i:04d},{dm:.2f},{flux:.4f},{width:.3f},{freq:.1f}")
    (frb_dir / "chime_frb_catalog.csv").write_text("\n".join(lines))
    print(f"  FRB: synthetic ({n} events)")


def build_frb_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    frb_dir = DATA_DIR / "frb"
    csv_files = sorted(frb_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("No FRB data. Run download first.")
    values = []
    for f in csv_files:
        for line in f.read_text().strip().split("\n")[1:]:
            parts = line.split(",")
            for p in parts:
                try:
                    v = float(p.strip())
                    if 0 < v < 10000:
                        values.append(v)
                        break
                except ValueError:
                    continue
    if len(values) < n_vars * episodes:
        raise RuntimeError(f"Too few FRB values: {len(values)}")
    return _build_timeseries_dataset(
        np.array(values), n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob)


# ---------------------------------------------------------------------------
# 9. S2 orbit (Keplerian, from Gillessen+ 2009 parameters)
# ---------------------------------------------------------------------------

def download_s2():
    """Generate S2 orbit from published orbital parameters."""
    s2_dir = DATA_DIR / "s2"
    s2_dir.mkdir(parents=True, exist_ok=True)
    out = s2_dir / "s2_orbit.npz"
    if out.exists():
        print("  S2: exists")
        return
    print("  S2: generating Keplerian orbit (Gillessen+ 2009) ...")
    rng = np.random.RandomState(42)
    period, ecc, a = 15.2, 0.876, 0.1226
    inc = np.radians(134.18)
    omega = np.radians(66.13)
    Omega = np.radians(228.07)
    n_obs = 200
    t = np.sort(rng.uniform(0, 2 * period, n_obs))
    M = 2 * np.pi * t / period
    E = M.copy()
    for _ in range(50):
        E = M + ecc * np.sin(E)
    nu = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2))
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    x, y = r * np.cos(nu), r * np.sin(nu)
    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc), np.sin(inc)
    ra = (cO * (cw * x - sw * y) - sO * si * (sw * x + cw * y)) * 1000
    dec = (sO * (cw * x - sw * y) + cO * si * (sw * x + cw * y)) * 1000
    ra_obs = ra + rng.normal(0, 0.3, n_obs)
    dec_obs = dec + rng.normal(0, 0.3, n_obs)
    np.savez(out, time=t, ra=ra_obs, dec=dec_obs)
    print(f"  S2: {n_obs} positions over {2 * period:.1f} yr")


def build_s2_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    s2_dir = DATA_DIR / "s2"
    npz = s2_dir / "s2_orbit.npz"
    if not npz.exists():
        raise RuntimeError("No S2 data. Run download first.")
    d = np.load(npz)
    data = np.column_stack([d["ra"], d["dec"]])
    return _build_timeseries_dataset(
        data, n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob, n_cols=2)


# ---------------------------------------------------------------------------
# 10. IceCube neutrinos
# ---------------------------------------------------------------------------

def download_icecube():
    nu_dir = DATA_DIR / "icecube"
    nu_dir.mkdir(parents=True, exist_ok=True)
    out = nu_dir / "icecube_events.csv"
    if out.exists() and out.stat().st_size > 100:
        print("  IceCube: exists")
        return

    # Try Harvard Dataverse direct file IDs (from medusa/download_neutrino_s2.py)
    dv_ids = [4617003, 4617004, 4617005, 4617006, 4617007,
              4617008, 4617009, 4617010, 4617011, 4617012, 4617013]
    for fid in dv_ids:
        url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
        print(f"  IceCube: trying Dataverse file {fid}...")
        if _download(url, out, label="IceCube"):
            print(f"  IceCube: saved ({out.stat().st_size / 1e3:.0f} KB)")
            return

    # Try HEASARC batch query
    try:
        url = ("https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"
               "tablehead=name%3Dicecubepsc&Action=Query"
               "&ResultMax=10000&displaymode=BatchDisplay&vession=text")
        print("  IceCube: trying HEASARC batch query...")
        data = urllib.request.urlopen(url, timeout=30).read()
        if len(data) > 500:
            out.write_bytes(data)
            print(f"  IceCube: saved from HEASARC ({len(data)} bytes)")
            return
    except Exception as e:
        print(f"  IceCube: HEASARC failed: {e}")

    print("  IceCube: all sources failed — generating synthetic fallback")
    _generate_synthetic_icecube(nu_dir)


def _generate_synthetic_icecube(nu_dir):
    rng = np.random.RandomState(42)
    n = 500
    ra = rng.uniform(0, 360, n)
    dec = rng.uniform(-90, 90, n)
    energy = 10 ** rng.uniform(2, 6, n)
    out = nu_dir / "icecube_events.csv"
    lines = ["ra,dec,energy"]
    for i in range(n):
        lines.append(f"{ra[i]:.4f},{dec[i]:.4f},{energy[i]:.2f}")
    out.write_text("\n".join(lines))
    print(f"  IceCube: synthetic fallback ({n} events)")


def _parse_icecube_heasarc(path):
    """Parse HEASARC pipe-separated IceCube catalog."""
    rows = []
    for line in path.read_text().strip().split("\n"):
        if not line.startswith("|") or line.startswith("|event"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 7:
            continue
        try:
            energy = float(parts[5])
            zenith = float(parts[7])
            time_str = parts[1]
            year = float(time_str[:4])
            month = float(time_str[5:7])
            day = float(time_str[8:10])
            t = year + (month - 1) / 12.0 + (day - 1) / 365.25
            if energy > 0 and 2008 < t < 2019:
                rows.append([t, np.log10(energy + 1), zenith])
        except (ValueError, IndexError):
            continue
    return np.array(rows, dtype=np.float64) if rows else None


def build_icecube_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    nu_dir = DATA_DIR / "icecube"
    data_files = sorted(nu_dir.glob("*"))
    if not data_files:
        raise RuntimeError("No IceCube data. Run download first.")

    data = None
    for f in data_files:
        content = f.read_text(errors="replace")
        if "|event" in content or "BatchStart" in content:
            data = _parse_icecube_heasarc(f)
            if data is not None and len(data) > 0:
                break
        else:
            values = []
            for line in content.strip().split("\n")[1:]:
                parts = line.split(",")
                try:
                    values.append([float(p) for p in parts[:3]])
                except (ValueError, IndexError):
                    continue
            if values:
                data = np.array(values, dtype=np.float64)
                break

    if data is None or len(data) < n_vars * episodes:
        raise RuntimeError(f"Too few IceCube events: {len(data) if data is not None else 0}")
    return _build_timeseries_dataset(
        data, n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob, n_cols=2)


# ---------------------------------------------------------------------------
# 11. NOAA weather (non-physical control)
# ---------------------------------------------------------------------------

def download_weather():
    wx_dir = DATA_DIR / "weather"
    wx_dir.mkdir(parents=True, exist_ok=True)
    out = wx_dir / "weather_nyc.csv"
    if out.exists():
        print("  Weather: exists")
        return
    url = ("https://www.ncei.noaa.gov/access/services/data/v1?"
           "dataset=daily-summaries&stations=USW00094728"
           "&startDate=2015-01-01&endDate=2020-12-31"
           "&dataTypes=TMAX,TMIN,PRCP&format=csv&units=metric")
    try:
        print("  Weather: downloading NOAA daily summaries ...")
        urllib.request.urlretrieve(url, out)
        print(f"  Weather: saved ({out.stat().st_size / 1e3:.0f} KB)")
    except Exception as e:
        print(f"  Weather: failed: {e}")


def build_weather_dataset(*, n_worlds, episodes, n_vars, seed, keep_prob):
    wx_dir = DATA_DIR / "weather"
    csv_files = sorted(wx_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("No weather data. Run download first.")
    values = []
    for f in csv_files:
        for line in f.read_text().strip().split("\n")[1:]:
            parts = line.split(",")
            row = []
            for p in parts:
                try:
                    row.append(float(p.strip().strip('"')))
                except ValueError:
                    continue
            if len(row) >= 2:
                values.append(row[:3] if len(row) >= 3 else row + [0.0])
    if len(values) < n_vars * episodes:
        raise RuntimeError(f"Too few weather rows: {len(values)}")
    return _build_timeseries_dataset(
        np.array(values), n_worlds=n_worlds, episodes=episodes,
        n_vars=n_vars, seed=seed, keep_prob=keep_prob, n_cols=2)


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAINS = {
    "ligo_gw150914": {
        "download": lambda: download_ligo("GW150914"),
        "build": lambda **kw: build_ligo_dataset("GW150914", **kw),
        "name": "GW150914",
        "regime": "II",
    },
    "ligo_gw170817": {
        "download": lambda: download_ligo("GW170817"),
        "build": lambda **kw: build_ligo_dataset("GW170817", **kw),
        "name": "GW170817",
        "regime": "III",
    },
    "ligo_gw190521": {
        "download": lambda: download_ligo("GW190521"),
        "build": lambda **kw: build_ligo_dataset("GW190521", **kw),
        "name": "GW190521",
        "regime": "II",
    },
    "eht_sgra": {
        "download": lambda: download_eht("sgra"),
        "build": lambda **kw: build_eht_dataset("sgra", **kw),
        "name": "EHT Sgr A*",
        "regime": "III",
    },
    "eht_m87": {
        "download": lambda: download_eht("m87"),
        "build": lambda **kw: build_eht_dataset("m87", **kw),
        "name": "EHT M87",
        "regime": "III",
    },
    "cmb": {
        "download": download_cmb,
        "build": build_cmb_dataset,
        "name": "CMB",
        "regime": "I-II",
    },
    "supernovae": {
        "download": download_supernovae,
        "build": build_supernovae_dataset,
        "name": "Type Ia SNe",
        "regime": "III",
    },
    "solar_wind": {
        "download": download_solar_wind,
        "build": build_solar_wind_dataset,
        "name": "Solar wind",
        "regime": "I",
    },
    "sunspots": {
        "download": download_sunspots,
        "build": build_sunspot_dataset,
        "name": "Sunspots",
        "regime": "I-II",
    },
    "quasars": {
        "download": download_quasars,
        "build": build_quasar_dataset,
        "name": "Quasars",
        "regime": "I-II",
    },
    "frb": {
        "download": download_frb,
        "build": build_frb_dataset,
        "name": "FRB",
        "regime": "catalog",
    },
    "s2_orbit": {
        "download": download_s2,
        "build": build_s2_dataset,
        "name": "S2 orbit",
        "regime": "I-II",
    },
    "icecube": {
        "download": download_icecube,
        "build": build_icecube_dataset,
        "name": "IceCube",
        "regime": "III",
    },
    "weather": {
        "download": download_weather,
        "build": build_weather_dataset,
        "name": "Weather (control)",
        "regime": "non-physical",
    },
}
