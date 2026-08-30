from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PopenFactory = Callable[..., subprocess.Popen[str]]


def run_benchmark_subprocess(
    request: dict[str, Any],
    timeout_seconds: float,
    *,
    popen_factory: PopenFactory = subprocess.Popen,
) -> dict[str, Any]:
    model = str(request["model"])
    with tempfile.TemporaryDirectory(prefix=".mry-benchmark-", dir=Path.cwd()) as directory:
        request_path = Path(directory) / "request.json"
        output_path = Path(directory) / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "mry_alert",
            "_benchmark-worker",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]
        environment = os.environ.copy()
        process = popen_factory(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            logger.error("Model %s exceeded %.1f second timeout", model, timeout_seconds)
            return {
                "model": model,
                "status": "timed_out",
                "error": f"model exceeded {timeout_seconds:g} second timeout",
                "segments": [],
                "worker_stdout": stdout[-4000:],
                "worker_stderr": stderr[-4000:],
            }
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return {"model": model, "status": "interrupted", "segments": []}
        if stdout:
            logger.info("Benchmark worker output for %s:\n%s", model, stdout.rstrip())
        if stderr:
            logger.info("Benchmark worker diagnostics for %s:\n%s", model, stderr.rstrip())
        if process.returncode != 0 or not output_path.exists():
            return {
                "model": model,
                "status": "failed",
                "error": stderr.strip() or f"worker exited with code {process.returncode}",
                "segments": [],
            }
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Benchmark worker returned a non-object result")
        return result
