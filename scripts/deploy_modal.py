import sys
import io
import os
import subprocess

# Force UTF-8 on Windows stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

cmd = [sys.executable, "-m", "modal", "deploy", "modal_app.py"]
print(f"Executing: {' '.join(cmd)}")
proc = subprocess.run(cmd, env=env)
sys.exit(proc.returncode)
