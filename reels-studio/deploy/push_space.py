"""Assemble the Reels Studio Space and push it to Hugging Face.

  HF_TOKEN=hf_xxx HF_SPACE=<user>/reels-studio STUDIO_ACCESS_KEY=<secret> python3 reels-studio/deploy/push_space.py

Creates the Docker Space if needed, stores STUDIO_ACCESS_KEY as a Space secret (the app then
only opens through ?key=<secret>), and uploads the build context. Needs `huggingface_hub`.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
PIPE = ["setup.sh", "requirements.txt", "requirements-gpu.txt", "reel", "reelkit", "library"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "data", ".models", ".tools", "fonts")


def main():
    token, space, key = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE"), os.environ.get("STUDIO_ACCESS_KEY")
    missing = [n for n, v in (("HF_TOKEN", token), ("HF_SPACE", space), ("STUDIO_ACCESS_KEY", key)) if not v]
    if missing:
        sys.exit(f"Set {', '.join(missing)} (see the docstring).")
    if "/" not in space:
        space = f"{HfApi(token=token).whoami()['name']}/{space}"
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space, "STUDIO_ACCESS_KEY", key)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copy(REPO / "reels-studio/deploy/Dockerfile", out / "Dockerfile")
        shutil.copy(REPO / "reels-studio/deploy/space_readme.md", out / "README.md")
        for name in PIPE:
            src = REPO / "reel-pipeline" / name
            dst = out / "reel-pipeline" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree(src, dst, ignore=SKIP) if src.is_dir() else shutil.copy2(src, dst))
        shutil.copytree(REPO / "reels-studio", out / "reels-studio", ignore=SKIP)
        api.upload_folder(folder_path=str(out), repo_id=space, repo_type="space",
                          commit_message="Deploy Reels Studio", delete_patterns=["*"])
    host = space.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    print(f"Deployed. Building at https://huggingface.co/spaces/{space}")
    print(f"Open: https://{host}.hf.space/?key=<your STUDIO_ACCESS_KEY>")


if __name__ == "__main__":
    main()
