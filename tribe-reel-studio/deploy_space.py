"""Upload the app to a private Hugging Face Space (run by GitHub Actions, or locally).

Env: HF_TOKEN (write token, required), SPACE_NAME (default tribe-reel-studio).
After the first deploy, in the Space's Settings tab: add the secret HF_TOKEN (a token
with LLaMA 3.2 access) and choose GPU hardware.
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

here = Path(__file__).resolve().parent


def fail(step, e):
    msg = f"{step} failed: {type(e).__name__}: {e}".replace("\n", " ")
    if "403" in msg or "401" in msg or "permission" in msg.lower():
        msg += " (Is HF_TOKEN a WRITE token?)"
    print(f"::error::{msg[:900]}")
    raise SystemExit(1)


api = HfApi(token=os.environ["HF_TOKEN"])
try:
    user = api.whoami()["name"]
except Exception as e:
    fail("Hugging Face login", e)
repo_id = f"{user}/{os.environ.get('SPACE_NAME') or 'tribe-reel-studio'}"

try:
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
except Exception as e:
    fail(f"Creating Space {repo_id}", e)
with tempfile.TemporaryDirectory() as tmp:
    for f in ("app.py", "brain_viz.py"):
        shutil.copy(here / f, tmp)
    for f in (here / "space").iterdir():
        shutil.copy(f, tmp)
    try:
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=tmp,
                          commit_message="Deploy from GitHub")
    except Exception as e:
        fail("Uploading app", e)

url = f"https://huggingface.co/spaces/{repo_id}"
print(f"\nApp URL: {url}")
print(f"::notice::App deployed: {url}")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(f"## 🧠 App URL: {url}\n\nIn the Space's Settings: add secret HF_TOKEN and pick a GPU.\n")
