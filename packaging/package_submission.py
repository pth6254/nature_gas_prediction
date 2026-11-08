"""Create a scoped submission copy; verify it in isolation; archive only selected files."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "deliverables" / "ineez_submission_20260906"
ARCHIVE = DEST.with_suffix(".zip")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection():
    files = {ROOT / name for name in ("README.md", "CLEANUP.md", "submission.csv", "pyproject.toml", "uv.lock", "REPORT.md", "(주)인이지+코딩테스트+안내.pdf")}
    for pattern in ("*data_train.csv", "*data_test.csv", "*submission_example.csv", "*metadata.xlsx"):
        found = list(ROOT.glob(pattern))
        assert len(found) == 1, (pattern, found)
        files.update(found)
    for folder in ("src", "tests"):
        files.update((ROOT / folder).glob("*.py"))
    files.update((ROOT / "notebooks").glob("*.md"))
    files.update(ROOT / "notebooks" / name for name in ("02_data_analysis.ipynb", "03_variable_suitability.ipynb"))
    for directory in ROOT.glob("outputs*"):
        if not directory.is_dir() or directory.name == "outputs_as_provided":
            continue
        for p in directory.iterdir():
            if not p.is_file():
                continue  # Exclude nested resumable caches.
            files.add(p)
    files.update((ROOT / "outputs_feature_audit/cache").glob("*.warnings.json"))
    for name in ("README.md", "REPORT_SLIDES.html", "slides.template.html", "build_deck.py", "evidence_manifest.json", "verify_deck.mjs", "qa_report.json"):
        files.add(ROOT / "presentation" / name)
    # Follow actual local links from the kept reports (including embedded figures).
    checked = set()
    while True:
        pending = [p for p in files - checked if p.suffix in (".md", ".html")]
        if not pending:
            break
        for p in pending:
            checked.add(p)
            content = p.read_text(encoding="utf-8-sig")
            targets = re.findall(r'\]\(([^)]+)\)|href="([^"]+)"', content)
            for pair in targets:
                target = next(x for x in pair if x).split("#")[0].strip("<>")
                if not target or re.match(r"^[a-z]+://", target) or target in {"REPORT_before_final_v1.md", "README.md"}:
                    continue
                resolved = (p.parent / target).resolve()
                if resolved.is_relative_to(ROOT) and resolved.is_file() and resolved.name != "README.md":
                    files.add(resolved)
    return sorted(files)


def prepare():
    if DEST.exists() or ARCHIVE.exists():
        raise FileExistsError("Submission destination already exists; refusing to overwrite")
    files = selection()
    DEST.mkdir(parents=True)
    for source in files:
        target = DEST / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run([sys.executable, "presentation/build_deck.py"], cwd=DEST, check=True)
    print(f"Prepared {DEST}; {len(files) + 1} files", flush=True)


def verify():
    assert DEST.is_dir()
    original_hash = digest(ROOT / "submission.csv")
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    assert digest(DEST / "submission.csv") == original_hash
    for p in (DEST / "src").glob("*.py"):
        assert digest(p) == digest(ROOT / "src" / p.name)
        ast.parse(p.read_text(encoding="utf-8"))
    # No source folder outside this isolated copy is placed on sys.path.
    scratch = Path(tempfile.mkdtemp(prefix="nature_gas_submission_verify_", dir="C:/Users/Public/Documents/ESTsoft/CreatorTemp"))
    work = scratch / "package"
    shutil.copytree(DEST, work)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(scratch / "matplotlib")
    results = []
    commands = [[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                [sys.executable, "-m", "src.final_pipeline"],
                [sys.executable, "-m", "src.final_report"],
                [sys.executable, "presentation/build_deck.py"]]
    for args in commands:
        print("Checking:", " ".join(args[1:]), flush=True)
        process = subprocess.run(args, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        results.append(dict(command="python " + " ".join(args[1:]), exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr))
        print(process.stdout[-1200:], flush=True)
        if process.returncode:
            (ROOT / "packaging/failed_verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(process.stderr[-3000:])
    assert digest(work / "submission.csv") == original_hash
    assert digest(ROOT / "submission.csv") == original_hash
    checks = dict(isolated_existing_environment=True, fresh_dependency_install_tested=False,
                  core_source_byte_identical=True, regenerated_submission_byte_identical=True,
                  original_submission_unchanged=True, historical_experiments_fully_rerun=False,
                  commands=results)
    (DEST / "PACKAGE_CHECKS.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Isolated verification passed.", flush=True)


def archive():
    checks = json.loads((DEST / "PACKAGE_CHECKS.json").read_text(encoding="utf-8"))
    assert checks["regenerated_submission_byte_identical"]
    checked_links = 0
    for name in ("README.md", "REPORT.md", "presentation/REPORT_SLIDES.html"):
        p = DEST / name
        for pair in re.findall(r'\]\(([^)]+)\)|href="([^"]+)"', p.read_text(encoding="utf-8")):
            target = next(x for x in pair if x).split("#")[0].strip("<>")
            if target and not re.match(r"^[a-z]+://", target):
                assert (p.parent / target).is_file(), (name, target)
                checked_links += 1
    slide_manifest = json.loads((DEST / "presentation/evidence_manifest.json").read_text(encoding="utf-8"))
    for name, expected in slide_manifest["sources"].items():
        assert digest(DEST / name) == expected, name
    checks["local_report_links_checked"] = checked_links
    checks["slide_source_hashes_match_package"] = True
    (DEST / "PACKAGE_CHECKS.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    assert not any(p.suffix == ".pyc" or any(x in p.parts for x in (".venv", "__pycache__", ".cache", "before_rubric_revision")) for p in files)
    manifest = {str(p.relative_to(DEST)).replace("\\", "/"): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files if p.name not in {"FILE_MANIFEST.json", "PACKAGE_CHECKS.json"}}
    (DEST / "FILE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DEST.parent))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for relative, info in manifest.items():
            assert hashlib.sha256(z.read(DEST.name + "/" + relative)).hexdigest() == info["sha256"]
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(manifest)+2, compressed_mb=round(ARCHIVE.stat().st_size/1024**2,2), sha256=digest(ARCHIVE)), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "verify", "archive"])
    {"prepare": prepare, "verify": verify, "archive": archive}[parser.parse_args().stage]()
