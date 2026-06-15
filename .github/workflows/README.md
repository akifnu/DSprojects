# GitHub Actions for TRIBE v2

## Free tier (CPU)

Workflow: [`tribev2-ci.yml`](tribev2-ci.yml)

Runs on `ubuntu-latest` at no cost for public repositories:

- Unit tests
- RCT dataset validation
- Environment checks

Triggers automatically on pushes and pull requests that touch `tribev2/`.

## GPU inference

Workflow: [`tribev2-framing-gpu.yml`](tribev2-framing-gpu.yml)

### Important limitations

1. **GitHub does not provide free GPU runners.** Standard `ubuntu-latest` runners have no GPU. GPU jobs use [larger runners](https://docs.github.com/en/actions/reference/runners/larger-runners) billed at **$0.052/minute** (Linux T4).
2. **GitHub GPU runners ship a Tesla T4 with 16 GB VRAM.** TRIBE v2 full trimodal text inference recommends **≥40 GB VRAM**. The workflow defaults to a **2-scenario smoke test** and may still OOM on T4.
3. **Larger runners require GitHub Team or Enterprise Cloud** and must be provisioned in your organization settings before the job can start.

### One-time setup

1. **Enable larger GPU runners** (org admin):
   - Organization → Settings → Actions → Runners → New runner → GPU → Linux
   - Note the runner **label** (e.g. `ubuntu-22.04-gpu`)
   - Add a spending limit under Billing

2. **Add repository secrets**:
   - `HF_TOKEN` — Hugging Face read token with access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

3. **Optional repository variable**:
   - `GPU_RUNNER_LABEL` — override the default `runs-on` label

### Run the GPU workflow

**Manual dispatch** (Actions tab → "TRIBE v2 Framing GPU" → Run workflow):

- Choose `max_scenarios` (1–8; start with 2 on T4)
- Confirm `runner_label` matches your org GPU runner

**Pull request label**:

- Apply the `run-gpu` label to a PR to trigger inference

### Artifacts

On completion, download `framing-rct-report` from the workflow run. It contains `framing_rct_analysis.json` with paired gain/loss statistics and Kahneman alignment metrics.

### If T4 runs out of memory

Use a machine with ≥40 GB VRAM locally:

```bash
cd tribev2
export HF_TOKEN=<token>
python scripts/run_framing_rct.py --preload-llama
```

Or provision a larger GPU runner if your GitHub plan supports custom sizes.
