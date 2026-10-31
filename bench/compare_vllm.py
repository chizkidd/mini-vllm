"""
Running vLLM next to the engine, on the same trace (Part of the head-to-head comparison; see bench/load.py for the traces and the metrics).

vLLM is a large package with its own pinned torch, so it lives in its own virtual environment and runs as its own OpenAI-compatible server
process; the load generator talks to it over HTTP exactly as it talks to this repo's server. Nothing here imports vllm.

  install_vllm(venv, version)    create the environment with uv and install vllm (a version, or None for the newest) into it
  vllm_versions(python)          the versions of vllm and torch inside the environment
  VllmServer(python, model, ...) start `vllm.entrypoints.openai.api_server`, wait until it answers /health, stop it (and its children)
  publish_results(...)           push only the named results files to the repo (the vLLM notebook is outside the stage chain, so it must never
                                 publish code)
  start_first_that_works(...)    try several environment settings in turn (a T4 is a Turing GPU: newer vLLM releases may need a different
                                 attention backend, or may not support it at all) and report every failed attempt with the tail of its log

Everything that can fail records why. A comparison that could not run vLLM says so in its results instead of leaving the column empty.
"""
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class VllmStartError(RuntimeError):
    def __init__(self, message, log_tail=""):
        super().__init__(message)
        self.log_tail = log_tail


def _tail(path, lines=40):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ---- the environment ----------------------------------------------------------------------------------------------

TORCH_BACKEND = "--torch-backend=auto"      # uv picks the PyTorch wheel index that matches the CUDA driver (vLLM's install page uses it)


def venv_python(venv):
    return str(Path(venv) / "bin" / "python")


def install_commands(venv, version=None, extra_pip=(), python="3.12"):
    """
    The commands that create `venv` and install vllm into it (a list of argv lists, run in order). The environment is made by uv, as
    vLLM's own install page does, and not by `python -m venv`: the system Python of a Kaggle image has no working ensurepip, so `venv`
    fails there before anything is installed. uv fetches the interpreter `python` when it is missing (None: the one running this).
    """
    spec = "vllm" if not version else f"vllm=={version}"
    uv = [sys.executable, "-m", "uv"]
    return [[sys.executable, "-m", "pip", "install", "--quiet", "uv"],
            [*uv, "venv", str(venv), *(["--python", python] if python else [])],
            [*uv, "pip", "install", "--python", venv_python(venv), TORCH_BACKEND, spec, *extra_pip]]


def install_vllm(venv, version=None, log_path="vllm_install.log", extra_pip=(), timeout_s=3600, run=subprocess.run):
    """
    Create the environment and install vllm. Returns {"ok", "python", "version_asked", "seconds", "error", "torch_backend"}; never raises for a
    failed install (the notebook records it and carries on without vLLM). The full install output goes to `log_path`. If the install fails
    with uv's torch-backend choice (the PyTorch index may be unreachable), it is retried once with the default torch wheels from PyPI, which
    are the ones vLLM is built against; `torch_backend` says which one produced the environment.
    """
    t0 = time.time()
    info = dict(ok=False, python=venv_python(venv), version_asked=version, seconds=None, error=None, torch_backend="auto")
    with open(log_path, "w") as log:
        for cmd in install_commands(venv, version, extra_pip):
            try:
                r = run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
                if r.returncode != 0 and TORCH_BACKEND in cmd:
                    log.write(f"\n[the install failed with {TORCH_BACKEND}; trying again with the default torch wheels]\n")
                    log.flush()
                    cmd, info["torch_backend"] = [c for c in cmd if c != TORCH_BACKEND], "default"
                    r = run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                info["error"] = f"timed out after {timeout_s} s running {' '.join(cmd[:4])} ..."
                break
            if r.returncode != 0:
                info["error"] = f"`{' '.join(cmd[:5])} ...` exited with {r.returncode}: {_tail(log_path, 12)}"
                break
        else:
            info["ok"] = True
    info["seconds"] = time.time() - t0
    return info


def vllm_versions(python, run=subprocess.run):
    """{"vllm": ..., "torch": ...} as seen from inside the environment, or {"error": ...}."""
    r = run([python, "-c", "import json, torch, vllm; print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__}))"],
            capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else f"exit {r.returncode}"}
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---- the server -----------------------------------------------------------------------------------------------------

def server_command(python, model, port, max_model_len=2048, gpu_memory_utilization=0.85, dtype="half", extra_args=()):
    """The command line of vLLM's OpenAI-compatible server."""
    return [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port), "--dtype", dtype,
            "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization), *extra_args]


class ServerProcess:
    """
    Any HTTP server run as a subprocess that answers GET /health when it is ready: vLLM's, or this repo's (`python -m minivllm.server`), so
    both are launched, waited for and stopped the same way and are driven by the same client. The process is started in its own session, so
    stop() ends the server and any worker processes it spawned.
    """
    def __init__(self, command, port, model="model", env=None, log_path="server.log"):
        self.command, self.model, self.port, self.log_path, self.env = list(command), model, port, log_path, env or {}
        self.proc = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _healthy(self):
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self, timeout_s=900, poll_s=1.0):
        """Launch the process and wait until /health answers. Raises VllmStartError (with the end of the log) if it exits or times out."""
        log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **self.env}, start_new_session=True)
        log.close()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            code = self.proc.poll()
            if code is not None:
                raise VllmStartError(f"the server exited with code {code} before it was ready", _tail(self.log_path))
            if self._healthy():
                self.ready_s = time.time() - t0
                return self
            time.sleep(poll_s)
        self.stop()
        raise VllmStartError(f"the server was not ready after {timeout_s} s", _tail(self.log_path))

    def stop(self, grace_s=20.0):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=grace_s)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait(timeout=grace_s)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class VllmServer(ServerProcess):
    """A vLLM server process. `command` replaces the default command line (the tests start a stand-in server that way)."""
    def __init__(self, python, model, port=8001, max_model_len=2048, gpu_memory_utilization=0.85, dtype="half", extra_args=(), env=None,
                 log_path="vllm_server.log", command=None):
        super().__init__(command or server_command(python, model, port, max_model_len, gpu_memory_utilization, dtype, extra_args), port, model, env, log_path)


def start_first_that_works(python, model, attempts, port=8001, timeout_s=900, log_dir=".", **server_kw):
    """
    Start a server with the first of `attempts` (dicts with a "name" and optional "env" and "extra_args") that comes up. Returns
    (server, record) where record lists every attempt: its name, whether it started, how long that took, and for failures the error and
    the end of the log. server is None if none worked.
    """
    record = []
    for i, a in enumerate(attempts):
        srv = VllmServer(python, model, port=port, env=a.get("env"), extra_args=a.get("extra_args", ()),
                         log_path=str(Path(log_dir) / f"vllm_server_{i}.log"), **server_kw)
        t0 = time.time()
        try:
            srv.start(timeout_s=timeout_s)
            record.append(dict(name=a["name"], ok=True, seconds=time.time() - t0, env=a.get("env", {}), extra_args=list(a.get("extra_args", ()))))
            return srv, record
        except VllmStartError as e:
            record.append(dict(name=a["name"], ok=False, seconds=time.time() - t0, error=str(e), log_tail=e.log_tail,
                               env=a.get("env", {}), extra_args=list(a.get("extra_args", ()))))
            srv.stop()
    return None, record


# ---- publishing -----------------------------------------------------------------------------------------------------

def publish_results(token, message, files, results_dir="results", repo=None, clone="mini-vllm-results", url=None, name=None, email=None):
    """
    Clone the repo (if needed), pull, copy ONLY the named files of `results_dir` into its results/ folder, commit and push. Unlike
    bench.publish.publish this never touches minivllm/ or bench/ or the stage stamp: a notebook outside the numbered series (this comparison)
    has no stage, and an old copy of the code must not overwrite newer code. `url` replaces the GitHub URL (the tests use a local bare repository).
    """
    import shutil
    from bench.publish import GIT_EMAIL, GIT_NAME, REPO, _run
    repo, name, email = repo or REPO, name or GIT_NAME, email or GIT_EMAIL
    clone = Path(clone)
    auth_url = url or f"https://{token}@github.com/{repo}.git"
    plain_url = url or f"https://github.com/{repo}.git"

    def git(*args):
        return _run(["git", *args], cwd=clone, secret=token, show=True)

    if not (clone / ".git").exists():
        _run(["git", "clone", "--progress", auth_url, str(clone)], secret=token, show=True)
        git("remote", "set-url", "origin", plain_url)          # do not keep the token in the clone's config
    git("config", "user.email", email)
    git("config", "user.name", name)
    git("pull", "--rebase", "--progress", auth_url, "HEAD")
    (clone / "results").mkdir(exist_ok=True)
    for f in files:
        shutil.copy2(Path(results_dir) / f, clone / "results" / f)
    git("add", "-A")
    if _run(["git", "status", "--porcelain"], cwd=clone, secret=token).strip():
        git("commit", "-m", message)
    else:
        print("nothing new to commit")
    git("push", "--progress", auth_url, "HEAD")
    return dict(pushed=True, files=list(files))
