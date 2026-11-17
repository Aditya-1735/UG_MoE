#!/usr/bin/env python3
"""
Download, verify and extract the Eadro SN / TT datasets from Zenodo (record 7615394).

    python -m data_pipeline.fetch_eadro                        # both systems, default root
    python -m data_pipeline.fetch_eadro --systems SN --root /path/to/datasets/eadro

Result: <root>/raw/<SN|TT>/{data,no fault}/... with every run directory (metrics/, logs.json,
spans.json) and its <system>.fault-<run>.json. Fault-free runs ship as .tar.xz and are extracted
next to their fault files. Archives are kept in <root>/archives/ and re-used if their MD5 matches.
"""
import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/7615394/files/{name}/content"
FILES = {
    "SN": ("SN Dataset.zip", "4c523faf64770b1e9a709cdd610e07c4"),
    "TT": ("TT Dataset.zip", "2d77861e4a025781e598d51866fd6f4e"),
}
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "eadro"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, expected_md5: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5(dest) == expected_md5:
        print(f"[ok] {dest.name} already present, MD5 verified")
        return dest
    url = RECORD.format(name=name.replace(" ", "%20"))
    print(f"[..] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    got = md5(tmp)
    if got != expected_md5:
        tmp.unlink()
        raise RuntimeError(f"MD5 mismatch for {name}: got {got}, expected {expected_md5}")
    tmp.replace(dest)
    print(f"[ok] {dest.name} downloaded, MD5 verified")
    return dest


def _safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise RuntimeError(f"Refusing to extract outside {base}: {member_name}")
    return target


def extract_zip(archive: Path, out: Path, top: str) -> None:
    """Extract '<top>/data/**' and '<top>/no fault/**' (skips __MACOSX, .DS_Store, *-all archives)."""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            n = info.filename
            if "__MACOSX" in n or n.endswith(".DS_Store") or not n.startswith(top + "/"):
                continue
            rel = n[len(top) + 1:]
            if not (rel.startswith("data/") or rel.startswith("no fault/")):
                continue  # e.g. SN-all.tgz / TT-all.txz: duplicate bundles of the same runs
            target = _safe_target(out, rel)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)


def extract_nested_tars(out: Path) -> None:
    """Fault-free runs are shipped as '<run>.tar.xz' inside 'no fault/'; extract them in place."""
    for tar_path in sorted((out / "no fault").glob("*.tar.xz")):
        run_dir = out / "no fault" / tar_path.name[: -len(".tar.xz")]
        if run_dir.exists():
            continue
        with tarfile.open(tar_path, "r:xz") as t:
            for m in t.getmembers():
                if m.issym() or m.islnk() or m.isdev():
                    continue  # never follow links / create devices from an archive
                _safe_target(out / "no fault", m.name)
                t.extract(m, out / "no fault")
        if not run_dir.exists():
            # some archives store files under a differently named top folder; report it
            print(f"[warn] {tar_path.name} did not produce {run_dir.name}/; check its layout")
        else:
            tar_path.unlink()
        print(f"[ok] extracted {tar_path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="SN,TT")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    for system in [s.strip() for s in args.systems.split(",")]:
        name, digest = FILES[system]
        archive = download(name, digest, root / "archives" / name)
        out = root / "raw" / system
        if (out / "data").exists():
            print(f"[ok] {out} already extracted")
        else:
            extract_zip(archive, out, top=name[: -len(".zip")])
        extract_nested_tars(out)
        runs = sorted(p.name for p in (out / "data").iterdir() if p.is_dir())
        clean = sorted(p.name for p in (out / "no fault").iterdir() if p.is_dir())
        print(f"[ok] {system}: {len(runs)} fault runs, {len(clean)} fault-free runs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
