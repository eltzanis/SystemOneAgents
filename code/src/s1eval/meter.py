"""Cost of each request, measured the same way for the LLM agents and the System 1 agent.

Per request (Meter.measure, around the agent's work for that request):
  seconds        wall-clock time
  gpu_energy_j   the GPU's own energy counter (NVML nvmlDeviceGetTotalEnergyConsumption, millijoules) read before and
                 after: the whole board, every process on it
  above_idle_j   gpu_energy_j minus the idle power (measured with the model loaded and no request, Meter.idle) x seconds
  power_w        mean of the power samples; gpu_util, gpu_util_max: mean and peak GPU utilisation (%); mem_used_mib:
                 peak memory used. Samples every 100 ms by a background thread (NVML)
  tokens         differences of llama-server's own counters (/metrics, the server runs with --metrics):
                 prompt_tokens = prompt tokens computed (not taken from the prompt cache), predicted_tokens = generated
  cpu_energy_j   the CPU package energy (RAPL, /sys/class/powercap/intel-rapl:0/energy_uj), only if this user may read
                 it (on Linux usually root only); otherwise None
Missing NVML or /metrics never stops a run: the fields are None and the reason is kept in Meter.problems. GPU energy is
the whole board's, so nothing else should run on the GPU while requests are measured.
"""
import ctypes
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

RAPL = Path("/sys/class/powercap/intel-rapl:0")


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class GPU:
    """The few NVML calls needed, through ctypes (no Python package)."""

    def __init__(self, index=0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._ok(self.lib.nvmlInit_v2())
        self.h = ctypes.c_void_p()
        self._ok(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.h)))
        name = ctypes.create_string_buffer(96)
        self._ok(self.lib.nvmlDeviceGetName(self.h, name, ctypes.c_uint(96)))
        self.name = name.value.decode()

    @staticmethod
    def _ok(code):
        if code != 0:
            raise RuntimeError(f"NVML error {code}")

    def energy_mj(self):
        e = ctypes.c_ulonglong()
        self._ok(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e)))
        return e.value

    def sample(self):
        u, p, m = _Util(), ctypes.c_uint(), _Mem()
        self._ok(self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)))
        self._ok(self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(p)))
        self._ok(self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)))
        return u.gpu, p.value / 1000.0, m.used / 2 ** 20


def rapl_uj():
    try:
        return int((RAPL / "energy_uj").read_text()), int((RAPL / "max_energy_range_uj").read_text())
    except (OSError, ValueError):
        return None


def server_counters(url):
    """llama-server's cumulative counters (/metrics, Prometheus text): {name: value} for every `llamacpp:*_total`."""
    txt = httpx.get(f"{url}/metrics", timeout=10).text
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^llamacpp:(\w+_total)\s+([0-9.eE+-]+)", txt, re.M)}


class Meter:
    def __init__(self, url=None, gpu_index=0, interval=0.1):
        self.url, self.interval, self.problems = url, interval, []
        self.idle_w = None
        try:
            self.gpu = GPU(gpu_index)
        except (OSError, RuntimeError, AttributeError) as e:
            self.gpu = None
            self.problems.append(f"no GPU meter: {e}")
        self.samples, self._lock, self._stop = [], threading.Lock(), threading.Event()
        if self.gpu:
            threading.Thread(target=self._run, daemon=True).start()
        if rapl_uj() is None:
            self.problems.append(f"no CPU energy: {RAPL}/energy_uj not readable")
        if url:
            try:
                server_counters(url)
            except (httpx.HTTPError, OSError) as e:
                self.problems.append(f"no token counters: {url}/metrics ({type(e).__name__}); start llama-server with --metrics")
                self.url = None

    def _run(self):
        while not self._stop.is_set():
            try:
                s = (time.perf_counter(), *self.gpu.sample())
                with self._lock:
                    self.samples.append(s)
                    if len(self.samples) > 200_000:
                        del self.samples[:100_000]
            except RuntimeError:
                pass
            time.sleep(self.interval)

    def close(self):
        self._stop.set()

    def _window(self, t0, t1):
        """The samples taken between t0 and t1; for a request shorter than the sampling interval, the latest sample."""
        with self._lock:
            w = [s for s in self.samples if t0 <= s[0] <= t1]
            return w or self.samples[-1:]

    def idle(self, seconds=20):
        """Mean power with nothing to do (the model loaded): the baseline subtracted for above_idle_j."""
        if not self.gpu:
            return None
        e0, t0 = self.gpu.energy_mj(), time.perf_counter()
        time.sleep(seconds)
        e1, t1 = self.gpu.energy_mj(), time.perf_counter()
        self.idle_w = (e1 - e0) / 1000.0 / (t1 - t0)
        w = self._window(t0, t1)
        return dict(idle_w=self.idle_w, idle_util=sum(s[1] for s in w) / len(w) if w else None, seconds=t1 - t0,
                    gpu=self.gpu.name)

    @contextmanager
    def measure(self):
        """with meter.measure() as cost: ...work...  -> cost (a dict) is filled when the block ends."""
        cost = {}
        c0 = self._counters()
        r0 = rapl_uj()
        e0 = self.gpu.energy_mj() if self.gpu else None
        t0 = time.perf_counter()
        try:
            yield cost
        finally:
            t1 = time.perf_counter()
            e1 = self.gpu.energy_mj() if self.gpu else None
            r1 = rapl_uj()
            c1 = self._counters()
            dt = t1 - t0
            cost["seconds"] = dt
            if e0 is not None:
                time.sleep(self.interval)                     # the last sample of the window
                w = self._window(t0, t1 + self.interval)
                cost["gpu_energy_j"] = (e1 - e0) / 1000.0
                cost["above_idle_j"] = cost["gpu_energy_j"] - self.idle_w * dt if self.idle_w is not None else None
                cost["power_w"] = sum(s[2] for s in w) / len(w) if w else None
                cost["gpu_util"] = sum(s[1] for s in w) / len(w) if w else None
                cost["gpu_util_max"] = max((s[1] for s in w), default=None)
                cost["mem_used_mib"] = max((s[3] for s in w), default=None)
            if r0 and r1:
                cost["cpu_energy_j"] = ((r1[0] - r0[0]) % r0[1]) / 1e6
            if c0 is not None and c1 is not None:
                d = {k: c1[k] - c0.get(k, 0.0) for k in c1}
                cost["prompt_tokens"] = d.get("prompt_tokens_total")
                cost["predicted_tokens"] = d.get("tokens_predicted_total")
                cost["counters"] = {k: v for k, v in d.items() if v}

    def _counters(self):
        if not self.url:
            return None
        try:
            return server_counters(self.url)
        except (httpx.HTTPError, OSError):
            return None
