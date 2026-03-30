#!/usr/bin/env python3
"""Start the OpenClaw-RL Combine LoRA training pipeline.

Launches ``openclaw-combine/run_qwen3_4b_openclaw_combine_lora.sh`` as a
background process so that subsequent notebook cells can continue without
waiting for the long-running training job to finish.

Output is tee'd to ``logs/training.log``.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
SCRIPT = REPO_ROOT / "openclaw-combine" / "run_qwen3_4b_openclaw_combine_lora.sh"
LOG_DIR = REPO_ROOT / "logs"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "training.log"

    env = os.environ.copy()
    env.setdefault("PORT", "30000")

    print(f"[start_combine_lora] Launching: {SCRIPT}", flush=True)
    print(f"[start_combine_lora] Log file : {log_path}", flush=True)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", str(SCRIPT)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )

    print(f"[start_combine_lora] Training process started (PID={proc.pid})", flush=True)
    print(f"[start_combine_lora] Monitor: tail -f {log_path}", flush=True)


if __name__ == "__main__":
    main()
