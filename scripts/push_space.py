"""Stage the files a Hugging Face Docker Space needs and (optionally) upload them.

    uv run --with huggingface-hub python scripts/push_space.py --dry-run        # stage only
    uv run --with huggingface-hub python scripts/push_space.py <user>/fraud-detection-api

Uses the token from `hf auth login` (or HF_TOKEN). The Space must already exist (Docker SDK).
upload_folder is used instead of git push because Spaces reject binary files (model.joblib)
pushed without LFS/Xet; the Hub client handles that automatically.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "uv.lock", ".python-version", "spaces/start.sh"]
DIRS = ["src", "api", "models", "demo"]


def stage(out: Path) -> Path:
    for f in FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in DIRS:
        shutil.copytree(ROOT / d, out / d, ignore=ignore)
    shutil.copy2(ROOT / "spaces/Dockerfile", out / "Dockerfile")
    shutil.copy2(ROOT / "spaces/README.md", out / "README.md")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", nargs="?", help="e.g. zeeqy/fraud-detection-api")
    ap.add_argument("--dry-run", action="store_true", help="stage to ./build/space and stop")
    args = ap.parse_args()

    if args.dry_run:
        out = ROOT / "build" / "space"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        stage(out)
        print(f"Staged Space files in {out}")
        return
    if not args.repo_id:
        ap.error("repo_id is required unless --dry-run")

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        folder = stage(Path(tmp))
        HfApi().upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=folder,
            commit_message="Deploy fraud detection demo",
        )
    print(f"Uploaded. Build log: https://huggingface.co/spaces/{args.repo_id}?logs=build")


if __name__ == "__main__":
    main()
