#!/usr/bin/env python3
"""hwprobe.py — фактическое железо, а не паспорт.

Чистый stdlib (никаких torch/numpy) — чтобы запускался где угодно, включая
«голый» контейнер и б/у рабочую станцию без CUDA.

Измеряет то, что реально определяет tok/s и Дж/токен при decode:
  1. DRAM-полоса (memcpy, 1..N потоков) — главный знаменатель в tok/s;
  2. доступная RAM (потолок для offload-экспертов и KV-кэша);
  3. GPU: имя, VRAM, полоса из bar1/nvidia-smi, текущая мощность, лимит, троттлинг;
  4. RAPL-мощность (если есть) — для Дж/токен без ваттметра;
  5. последовательное чтение диска (для сценария «эксперты с NVMe»).

Использование:
    python3 tools/hwprobe.py                 # человекочитаемо
    python3 tools/hwprobe.py --json out/hw.json
    python3 tools/hwprobe.py --mb 512 --threads 4 --rapl-seconds 5
    python3 tools/aira_calc.py route --probe out/hw.json     # что делать дальше
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from multiprocessing import Pool

# ---------------------------------------------------------------- утилиты ---


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:
        return ""


def _num(s: str, unit: float = 1.0, default: float = 0.0) -> float:
    m = re.search(r"[-+]?\d*\.?\d+", s or "")
    return float(m.group()) * unit if m else default


# ------------------------------------------------------------------- CPU ---


def probe_cpu() -> dict:
    info = {"cores_logical": os.cpu_count(), "machine": platform.machine(),
            "os": f"{platform.system()} {platform.release()}"}
    ci = _read("/proc/cpuinfo")
    for line in ci.splitlines():
        if line.lower().startswith("model name"):
            info["model"] = line.split(":", 1)[1].strip()
            break
    if "model" not in info and platform.system() == "Darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    # частота (если доступна из sysfs)
    f = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
    if f:
        info["cpufreq_governor"] = f
    if platform.system() == "Linux" and ci:
        mhz = [float(x) for x in re.findall(r"cpu MHz\s*:\s*([\d.]+)", ci)]
        if mhz:
            info["mhz_avg"] = round(sum(mhz) / len(mhz), 1)
    return info


def probe_ram() -> dict:
    out = {}
    mi = _read("/proc/meminfo")
    for key, kbytes in (("MemTotal", 1), ("MemAvailable", 1), ("SwapTotal", 1)):
        m = re.search(rf"^{key}:\s+(\d+)", mi, re.M)
        if m:
            out[key] = round(int(m.group(1)) / 1048576.0, 2)  # GiB
    if platform.system() == "Darwin":
        phys = _run(["sysctl", "-n", "hw.memsize"])
        n = _num(phys)
        if n:
            out["MemTotal"] = round(n / 1073741824.0, 2)
    # size of transparent hugepages влияет на memcpy-результаты; полезно знать
    thp = _read("/sys/kernel/mm/transparent_hugepage/enabled").strip()
    if thp:
        out["thp"] = thp
    return out


# --------------------------------------------------- полоса памяти (DRAM) ---


def _memcpy_once(args) -> float:
    """Один поток: сколько GiB/s копирует.  Возвращает GB/s (read+write)."""
    mb, iters = args
    size = mb * 1024 * 1024
    src = bytearray(size)
    dst = bytearray(size)
    mv_src, mv_dst = memoryview(src), memoryview(dst)
    t0 = time.perf_counter()
    done = 0
    while done < iters:
        mv_dst[:] = mv_src           # C-level memmove: read + write
        done += 1
    dt = (time.perf_counter() - t0) / max(1, done)
    del src, dst
    return (2.0 * size) / dt / 1e9   # байт скопировано + прочитано


def probe_bandwidth(mb: int, threads: int, iters: int = 3) -> dict:
    single = min(_memcpy_once((mb, iters)) for _ in range(max(1, min(iters, 2))))
    res = {"buffer_MB": mb, "single_thread_GBs": round(single, 1)}
    if threads > 1:
        try:
            with Pool(threads) as pool:
                vals = pool.map(_memcpy_once, [(mb, iters)] * threads)
            res[f"threads_{threads}_GBs"] = round(sum(vals), 1)
        except Exception as exc:  # noqa: BLE001
            res["threads_error"] = str(exc)
    res["note"] = ("порядок: single<threads значит ещё не занятая полоса; "
                   "для оценки tok/s используйте 0.55-0.75 от aggregate значения")
    return res


# ------------------------------------------------------------------- GPU ---


def probe_gpu() -> dict:
    out = {}
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,power.draw,"
                "power.limit,clocks.max.mem,temperature.gpu,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits"])
    if smi.strip():
        gpus = []
        for line in smi.strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            if len(c) >= 8:
                gpus.append({
                    "name": c[0], "vram_GB": _num(c[1], 1 / 1024.0),
                    "mem_used_GB": _num(c[2], 1 / 1024.0),
                    "power_W": _num(c[3]), "power_limit_W": _num(c[4]),
                    "mem_clock_max_MHz": _num(c[5]), "temp_C": _num(c[6]),
                    "throttle": c[7],
                })
        out["nvidia"] = gpus
        # пропускная способность: оценим из clock*шины; уточнить по datasheet
        bus = _run(["nvidia-smi", "--query-gpu=memory.bus.width", "--format=csv,noheader"])
        if bus.strip():
            for g, b in zip(out["nvidia"], bus.strip().splitlines()):
                try:
                    g["mem_bus_bits_est"] = int(float(b))
                    g["bw_GBs_est"] = round(2 * g["mem_clock_max_MHz"] * 1e6 *
                                            int(float(b)) / 8 / 1e9, 0)
                except Exception:  # noqa: BLE001
                    pass
    rocm = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if rocm.strip():
        out["amd_rocm_raw"] = rocm.strip()[:400]
    if not out:
        out["none"] = ("нет nvidia-smi/rocm-smi. На macOS проверьте hw.ncpu, unified memory "
                       "и `sudo powermetrics` вручную")
    return out


# ---------------------------------------------------------------- RAPL ----


def rapl_paths() -> list[str]:
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return []
    return sorted(os.path.join(base, d, "energy_uj") for d in os.listdir(base)
                  if re.match(r"intel-rapl:\d+$", d))


def probe_power_sample(seconds: float) -> dict:
    """Средняя мощность по RAPL во время нагрузки. Только CPU/socket, без dGPU."""
    ps = rapl_paths()
    if not ps:
        return {"available": False}
    def snap():
        vals = []
        for p in ps:
            try:
                vals.append(int(_read(p).strip() or 0))
            except Exception:  # noqa: BLE001
                vals.append(0)
        return vals
    a, t0 = snap(), time.perf_counter()
    # фоновая нагрузка, чтобы измерение не было «на холостом»
    busy = Pool(2)
    [busy.apply_async(_memcpy_once, ((64, 4000),)) for _ in range(4)]
    time.sleep(max(1.0, seconds))
    b, dt = snap(), time.perf_counter() - t0
    busy.terminate()
    joules = sum(max(0, y - x) for x, y in zip(a, b)) / 1e6
    return {"available": True, "sensors": len(ps), "W": round(joules / dt, 1),
            "note": "socket power during memcpy load (CPU only)"}


# ------------------------------------------------------------------ диск ---


def probe_disk(mb: int, path: str = "/tmp") -> dict:
    """Последовательные запись/чтение большого файла. Чтение — после вытеснения
    из page cache (posix_fadvise DONTNEED), иначе измеряется RAM."""
    f = os.path.join(path, f".hwprobe_{os.getpid()}.bin")
    res = {}
    try:
        size = mb * 1024 * 1024
        chunk = bytes(8 * 1024 * 1024)
        t0 = time.perf_counter()
        with open(f, "wb") as fh:
            for _ in range(size // len(chunk)):
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        res["write_GBs"] = round(size / (time.perf_counter() - t0) / 1e9, 2)
        # вытеснить из кэша и читать заново
        fd = os.open(f, os.O_RDONLY)
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            t0 = time.perf_counter()
            seen = 0
            while True:
                b = os.read(fd, 4 * 1024 * 1024)
                if not b:
                    break
                seen += len(b)
            res["read_GBs"] = round(seen / (time.perf_counter() - t0) / 1e9, 2)
        finally:
            os.close(fd)
    except Exception as exc:  # noqa: BLE001
        res["error"] = str(exc)
    finally:
        try:
            os.remove(f)
        except Exception:  # noqa: BLE001
            pass
    t = shutil.disk_usage(path)
    res["free_GB"] = round(t.free / 1e9, 1)
    res["note"] = ("если read ≈ write — скорее всего кэш не вытеснился (нет прав) "
                   "или это RAM-диски/tmpfs")
    if path.startswith("/tmp") and _read("/proc/mounts").find(" /tmp tmpfs") >= 0:
        res["warning"] = "/tmp = tmpfs: измеряется память, а не NVMe"
    return res


# ---------------------------------------------------------------- main -----


def main() -> int:
    ap = argparse.ArgumentParser(description="probe real hardware (stdlib only)")
    ap.add_argument("--mb", type=int, default=192, help="размер буфера для теста полосы, MiB")
    ap.add_argument("--threads", type=int, default=0, help="потоки (0 = все ядра)")
    ap.add_argument("--disk-mb", type=int, default=1024, help="0 = пропустить диск")
    ap.add_argument("--disk-path", default="/tmp")
    ap.add_argument("--rapl-seconds", type=float, default=0.0, help="0 = не мерить мощность")
    ap.add_argument("--json", dest="json_path", help="куда положить JSON")
    a = ap.parse_args()

    th = a.threads or max(1, (os.cpu_count() or 2))
    data = {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "cpu": probe_cpu(),
        "ram_GiB": probe_ram(),
        "bandwidth": probe_bandwidth(a.mb, min(th, 8)),
        "gpu": probe_gpu(),
        "rapl": probe_power_sample(a.rapl_seconds) if a.rapl_seconds else {"skipped": True},
    }
    if a.disk_mb:
        data["disk"] = probe_disk(a.disk_mb, a.disk_path)

    g = data["gpu"]
    bw = data["bandwidth"]
    print(f"# AIra hwprobe @ {data['host']}")
    print(f"CPU   : {data['cpu'].get('model','?')} · {data['cpu']['cores_logical']} потоков"
          f" · {data['cpu'].get('mhz_avg','?')} MHz")
    print(f"RAM   : {data['ram_GiB'].get('MemTotal','?')} GiB всего,"
          f" {data['ram_GiB'].get('MemAvailable','?')} GiB доступно")
    print(f"DRAM  : {bw['single_thread_GBs']} GB/s на поток"
          + (f" · {bw.get(f'threads_{min(th,8)}_GBs','?')} GB/s на {min(th,8)} потоках"))
    if "nvidia" in g and g["nvidia"]:
        for i, x in enumerate(g["nvidia"]):
            print(f"GPU{i}: {x['name']} · {x['vram_GB']:.0f} GiB "
                  f"(занято {x['mem_used_GB']:.1f}) · {x['power_W']:.0f}Вт "
                  f"/ лимит {x['power_limit_W']:.0f}Вт · {x.get('bw_GBs_est','?')} GB/s (оценка)")
    else:
        print("GPU   : дискретного CUDA/ROCm не видно -> CPU/unified-memory режим")
    if "disk" in data:
        d = data["disk"]
        print(f"DISK  : read {d.get('read_GBs','?')} / write {d.get('write_GBs','?')} GB/s"
              f" · свободно {d.get('free_GB','?')} GB")
    if data["rapl"].get("available"):
        print(f"POWER : ~{data['rapl']['W']} Вт (socket, под нагрузкой; RAPL)")
    print("\nДальше:  python3 tools/aira_calc.py route --probe "
          + (a.json_path or "out/hw.json"))

    if a.json_path:
        os.makedirs(os.path.dirname(a.json_path) or ".", exist_ok=True)
        with open(a.json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"записано: {a.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
