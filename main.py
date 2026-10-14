import asyncio
import glob
import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.parse
from typing import List, Optional
import cv2
import numpy as np
import openvino as ov
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

# Force TCP transport for RTSP pulls (far more reliable than UDP for sustained streams).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# Unbuffered logs so container output is visible immediately.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Silence FFmpeg/libav stderr noise (h264 SEI / decode warnings) so app logs stay clean.
# OpenCV ships its OWN bundled FFmpeg libs, so we must silence that instance too.
def _silence_ffmpeg():
    import ctypes
    import ctypes.util
    import glob
    import os
    import site
    import sys
    candidates = []
    for root in list(site.getsitepackages()) + list(sys.path):
        try:
            candidates += glob.glob(os.path.join(root, "opencv_python*.libs", "libavutil*.so*"))
        except Exception:
            pass
    sys_lib = ctypes.util.find_library("avutil")
    if sys_lib:
        candidates.append(sys_lib)
    silenced = []
    for path in candidates:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL).av_log_set_level(-8)  # AV_LOG_QUIET
            silenced.append(os.path.basename(path))
        except Exception:
            continue
    print(f"[startup] ffmpeg quiet on: {silenced}", flush=True)

_silence_ffmpeg()

def _install_stderr_filter():
    """Strip FFmpeg/libav messages (e.g. "[h264 @ 0x..] SEI type .. truncated ..") from
    stderr, keeping every other line (uvicorn logs, tracebacks). Configurable:
    set HALO_FFMPEG_QUIET=0 to disable. Our own app logs go to stdout and are unaffected."""
    if os.environ.get("HALO_FFMPEG_QUIET", "1") != "1":
        return
    import re
    import threading
    try:
        pat = re.compile(rb"\[[A-Za-z0-9_.]+ @ 0x[0-9a-fA-F]+\][^\[]*")
        r, w = os.pipe()
        saved = os.dup(2)
        os.dup2(w, 2)
        os.close(w)

        def relay():
            while True:
                try:
                    chunk = os.read(r, 8192)
                except Exception:
                    break
                if not chunk:
                    break
                cleaned = pat.sub(b"", chunk)
                if cleaned:
                    try:
                        os.write(saved, cleaned)
                    except Exception:
                        pass

        threading.Thread(target=relay, daemon=True, name="stderr-filter").start()
    except Exception:
        pass

_install_stderr_filter()

app = FastAPI(title="Halloween Automation Engine")

MODEL_NAME = "yolo26m"
MODEL_IMGSZ = 640  # 640x640 FP16 OpenVINO IR
# Activation gate: an output's trigger point is only satisfied when a detected
# position is at/past the point BUT still within this many normalized points of
# it. This stops outputs (especially Main) firing from far past the trigger point.
ACTIVATION_BAND = 20.0
# Paths are overridable for local/dev runs (defaults target the mounted /data volume).
CONFIG_PATH = os.environ.get("HALO_CONFIG", "/data/config.yml")
MODEL_DIR = os.environ.get("HALO_MODEL_DIR", f"/data/models/{MODEL_NAME}")

# --- DEFAULT CONFIGURATION & STATE ---
DEFAULT_CONFIG = {
    "device_priority": "NPU,GPU,CPU",  # accelerator preference order (CPU always last)
    # Inference rate cap. Higher = lower end-to-end latency (fresher positions), so set this
    # as high as the accelerator can sustain; the capture thread already drops stale frames.
    "max_fps": 16,
    "conf_threshold": 0.55,        # min person confidence to consider a detection
    "iou_match_threshold": 0.15,   # min box overlap (IoU) to treat as the same person across frames
    "min_consecutive_frames": 2,   # consecutive hits required to register a track
    "subsequent_conf_threshold": 0.40,  # lower conf allowed for an already-confirmed (tracked) person
    "track_validity_ms": 800,      # keep coasting (predicted) a track this long without a detection
    # Position output is the Kalman-filtered box. These two knobs tune latency vs steadiness:
    # position_smoothing: 0 = near-real-time / lowest latency (raw-ish, some jitter) .. 1 = heavy
    #   smoothing (steady dot, slightly laggy). Maps to the filter's measurement noise (R).
    "position_smoothing": 0.10,
    # track_agility: how quickly the filter follows sudden speed changes. Higher = the reported
    #   position stays locked to fast/quick movement. Maps to the velocity process noise (Q).
    "track_agility": 0.6,
    "auto_start": False,  # if False, inference waits until started from the UI
    # Site/network settings are intentionally blank -- fill them in on /setup.
    "rtsp_url": "",                # e.g. rtsp://user:pass@host:554/path
    "esp32_host": "",              # ESP32 GPIO controller IP (blank = auto-discover)
    "esp32_udp_port": 5001,
    "esp32_tcp_port": 5000,
    "output_names": {f"out_{i}": f"Output {i}" for i in range(1, 9)},
    "inclusion_polygon": [[41.5, 53.1], [43.4, 75.3], [84.4, 72.8], [83.9, 55.2]],  # [[x,y] x4] normalized 0-100
    "exclusion_polygons": [],  # [[[x,y], ...], ...]
    "base_line": {"p1": [44.7, 63.8], "p2": [81.4, 63.2]},
    "threshold_point": 50.0,
    "threshold_delay": 5.0,  # seconds after the threshold cross before reverse logic activates
    "outputs": {
        "1": {"trigger": 15.0, "duration": 180, "depends": False, "max_act": 1, "reverse": False},
        "2": {"trigger": 40.0, "duration": 8, "depends": True, "max_act": 1, "reverse": False},
        "3": {"trigger": 60.0, "duration": 0.5, "depends": True, "max_act": 1, "reverse": False},
        "4": {"trigger": 70.0, "duration": 0.5, "depends": True, "max_act": 1, "reverse": False},
        "5": {"trigger": 80.0, "duration": 0.5, "depends": True, "max_act": 1, "reverse": False},
        "6": {"trigger": 85.0, "duration": 0.5, "depends": True, "max_act": 1, "reverse": False},
        "7": {"trigger": 90.0, "duration": 0.5, "depends": True, "max_act": 1, "reverse": False},
        "8": {"trigger": 20.0, "duration": 0.5, "depends": True, "max_act": 1, "reverse": True},
    }
}

def load_config():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    return cfg
        except Exception:
            pass
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

config = load_config()

# --- RUNTIME STATE ---
class RuntimeState:
    def __init__(self):
        self.esp32_ip = None
        self.connection_status = "DISCONNECTED"
        self.last_command = "None"
        self.last_response = "None"
        self.detected_persons = []
        self.tracks = []            # [{id, position}] stable per-track ids for the UI
        self.inference_latency_ms = 0.0
        self.output_states = {str(i): 0 for i in range(1, 9)}
        self.output_timers = {str(i): 0.0 for i in range(1, 9)}
        self.output_activations_count = {str(i): 0 for i in range(1, 9)}
        self.threshold_crossed_at = None  # timestamp when anyone first reached the threshold point
        self.main_active = False
        self.model_loaded = False
        self.model_device = "UNINITIALIZED"
        self.model_input_size = (MODEL_IMGSZ, MODEL_IMGSZ)
        self.last_inference_ms = 0.0
        self.inference_ms_avg = 0.0
        self.inference_fps = 0.0
        self.stream_fps = 0.0
        self.frame_age_ms = 0.0     # how old the newest frame was when inference started
        self.frame_lag = 0.0        # frame_age_ms expressed in stream-frame units
        self.frames_skipped = 0     # captured frames dropped because inference was still busy
        self.engine_running = False
        self.device_warning = ""
        self.reverse_armed = False        # latch: reverse outputs armed (Main active + all past threshold)
        self.prev_max_position = None     # previous frame's highest position (for reverse crossing)

state = RuntimeState()

# --- OPENVINO MODEL (640x640 FP16 IR, NPU -> GPU -> CPU) ---
ov_compiled_model = None
ov_input_layer = None

def _find_weights():
    """Locate the YOLO26m .pt weights used to (re)build the OpenVINO IR."""
    candidates = []
    env = os.environ.get("HALO_WEIGHTS")
    if env:
        candidates.append(env)
    candidates += [
        os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt"),
        os.path.join("/app", f"{MODEL_NAME}.pt"),
        os.path.join("/models", f"{MODEL_NAME}.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{MODEL_NAME}.pt"),
        f"{MODEL_NAME}.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def build_openvino_model():
    """Export YOLO26m to a 640x640 FP16 OpenVINO IR (no e2e/Top-K nodes).

    ONE model is built and shared by every accelerator (NPU/GPU/CPU). The raw-head FP16 IR
    is the fastest option on all three, so do NOT fork a per-device build: a GPU/e2e variant
    (Top-K/NMS in-graph) measured ~1.4x slower on the Arc GPU than this raw-head IR, and the
    CPU postprocess it would remove costs ~0.01 ms/frame. See build_model.py for the CLI.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    weights = _find_weights()
    if not weights:
        print(f"[model] Weights '{MODEL_NAME}.pt' not found; set HALO_WEIGHTS or drop it in {MODEL_DIR}.")
        return False
    print(f"[model] Building {MODEL_NAME} OpenVINO IR ({MODEL_IMGSZ}x{MODEL_IMGSZ}, FP16, end2end=False)...")
    from ultralytics import YOLO  # lazy import: torch is only needed to (re)build the model
    model = YOLO(weights)
    export_path = model.export(
        format="openvino",
        imgsz=MODEL_IMGSZ,
        dynamic=False,
        quantize=16,     # FP16 weights (compress_to_fp16)
        batch=1,
        nms=False,       # keep raw logits (no embedded NMS)
        end2end=False,   # strip the YOLO26 Top-K/Gather/NMS head -> NPU friendly
    )
    src_dir = os.path.dirname(str(export_path))
    for ext in (".xml", ".bin"):
        src = os.path.join(src_dir, f"{MODEL_NAME}{ext}")
        if not os.path.exists(src):
            matches = glob.glob(os.path.join(src_dir, f"*{ext}"))
            src = matches[0] if matches else src
        dst = os.path.join(MODEL_DIR, f"{MODEL_NAME}{ext}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
    xml_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.xml")
    bin_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.bin")
    print(f"[model] Installed IR -> {xml_path}")
    return os.path.exists(xml_path) and os.path.exists(bin_path)

DEVICE_WARN_MSG = (
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "!!  WARNING: NO INTEL NPU/GPU DETECTED  --  RUNNING ON CPU               !!\n"
    "!!  Inference will be far slower and may not keep up with the stream.    !!\n"
    "!!  Fix: install the Level Zero NPU runtime (intel-level-zero-npu) and   !!\n"
    "!!  expose /dev/accel (NPU) and /dev/dri (GPU) to the container.         !!\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
)

def device_priority_order():
    """Parsed accelerator preference order from config (CPU is always appended last)."""
    raw = config.get("device_priority", "NPU,GPU,CPU")
    if isinstance(raw, list):
        order = [str(d).strip().upper() for d in raw]
    else:
        order = [d.strip().upper() for d in str(raw).split(",")]
    order = [d for d in order if d]
    if not order:
        order = ["NPU", "GPU", "CPU"]
    if "CPU" not in order:
        order.append("CPU")
    return order

def resolve_device(core, order):
    """Return the first device in `order` that the runtime exposes (CPU is the last resort)."""
    available = list(core.available_devices)
    for dev in order:
        if dev == "CPU":
            return "CPU"
        if any(a == dev or a.split(".")[0] == dev for a in available):
            return dev
    return "CPU"

def initialize_openvino_model():
    global ov_compiled_model, ov_input_layer
    os.makedirs(MODEL_DIR, exist_ok=True)
    xml_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.xml")
    bin_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.bin")

    if not (os.path.exists(xml_path) and os.path.exists(bin_path)):
        try:
            build_openvino_model()
        except Exception as e:
            print(f"[model] Build failed: {e}")

    if not os.path.exists(xml_path):
        print("[model] OpenVINO IR unavailable; inference disabled until the model is present.")
        ov_compiled_model = None
        state.model_loaded = False
        state.model_device = "NONE"
        return

    core = ov.Core()
    order = device_priority_order()
    device = resolve_device(core, order)
    print(f"[model] device priority={order} | available={core.available_devices} | selected={device}", flush=True)
    try:
        model_ov = core.read_model(model=xml_path, weights=bin_path)
        ov_compiled_model = core.compile_model(model=model_ov, device_name=device)
    except Exception as e:
        print(f"[model] Compile on {device} failed ({e}); retrying on CPU.", flush=True)
        device = "CPU"
        model_ov = core.read_model(model=xml_path, weights=bin_path)
        ov_compiled_model = core.compile_model(model=model_ov, device_name=device)

    if device == "CPU" and order[0] != "CPU":
        state.device_warning = DEVICE_WARN_MSG
        print(DEVICE_WARN_MSG, flush=True)
    else:
        state.device_warning = ""

    ov_input_layer = ov_compiled_model.input(0)
    try:
        shape = list(ov_input_layer.shape)
        state.model_input_size = (int(shape[3]), int(shape[2]))
    except Exception:
        state.model_input_size = (MODEL_IMGSZ, MODEL_IMGSZ)
    state.model_device = device
    state.model_loaded = True
    print(f"[model] Loaded {MODEL_NAME} on {device} | input {state.model_input_size} | devices={core.available_devices}")

# --- ESP32 NETWORK MANAGER ---
# A basic (non-S3) ESP32 runs a WiFiServer whose command loop BLOCKS while a client is
# connected and can only serve ONE client at a time. Overlapping connections -- or a
# leaked / half-open socket -- will wedge it. So every exchange here is serialized through
# a single lock, opens at most one short-lived connection, and is guaranteed to close it
# (bounded by socket timeouts, never left open on error).
_esp32_lock = asyncio.Lock()
_esp32_last_ok = 0.0              # wall-clock time of the last successful exchange
# GPIO commands are queued here and drained by ``esp32_dispatch_loop`` on a SEPARATE
# task, so the inference/FSM loop never blocks waiting on the (slow) controller network.
_esp32_cmd_queue: "asyncio.Queue[str]" = asyncio.Queue()
ESP32_CONNECT_TIMEOUT = 2.5       # seconds to establish the TCP connection
ESP32_POLL_TIMEOUT = 0.3          # per-recv poll slice (keeps draining until the batch deadline)
ESP32_BATCH_TIMEOUT = 4.0         # total seconds to collect a batch's replies (weak boards are slow)
ESP32_ATTEMPTS = 2                # connection attempts per exchange (commands are idempotent)
ESP32_HEALTH_INTERVAL = 30.0      # seconds between idle keep-alive PINGs

def _discover_esp32_once():
    """Blocking UDP broadcast probe (run in a thread so it never stalls the event loop)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.5)
        port = int(config.get("esp32_udp_port", 5001))
        targets = {"255.255.255.255"}
        try:  # also try the RTSP host's subnet broadcast (best effort)
            host = urllib.parse.urlsplit(config["rtsp_url"]).hostname or ""
            parts = host.split(".")
            if len(parts) == 4:
                targets.add(".".join(parts[:3]) + ".255")
        except Exception:
            pass
        for dst in targets:
            try:
                sock.sendto(b"DISCOVER_ESP32", (dst, port))
            except Exception:
                pass
        data, addr = sock.recvfrom(1024)
        return data.decode(errors="ignore"), addr[0]
    finally:
        sock.close()

async def esp32_discovery_loop():
    while True:
        host = (config.get("esp32_host") or "").strip()
        if host:
            # Manual IP: unicast works through the Docker bridge (broadcast does not).
            # Only set a provisional status here; the health loop owns the real
            # CONNECTED/ERROR state once it has actually talked to the controller.
            state.esp32_ip = host
            if state.connection_status in ("DISCONNECTED", "NO IP", "", "SEARCHING..."):
                state.connection_status = f"CONNECTED ({host})"
            await asyncio.sleep(5.0)
            continue
        try:
            payload, ip = await asyncio.to_thread(_discover_esp32_once)
            if "ESP32_ACK" in payload:
                state.esp32_ip = ip
                state.connection_status = f"CONNECTED ({ip})"
            elif not state.esp32_ip:
                state.connection_status = "SEARCHING..."
        except Exception:
            if not state.esp32_ip:
                state.connection_status = "SEARCHING..."
        await asyncio.sleep(5.0)

def _esp32_open():
    """Open ONE TCP connection to the controller with sane options and a bounded timeout."""
    port = int(config.get("esp32_tcp_port", 5000))
    sock = socket.create_connection((state.esp32_ip, port), timeout=ESP32_CONNECT_TIMEOUT)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # no Nagle delay
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)   # detect dead peers
    except Exception:
        pass
    sock.settimeout(ESP32_CONNECT_TIMEOUT)   # used for the send phase
    return sock

def _esp32_exchange_sync(commands):
    """Blocking: pipeline all commands over ONE short-lived connection and read one reply
    line per command. Every path is bounded by socket timeouts and the socket is ALWAYS
    closed (shutdown + close), so we never leak a half-open connection to the ESP32."""
    cmds = [c.rstrip("\n") + "\n" for c in commands]
    want = len(cmds)
    last_err = "no response"
    for attempt in range(ESP32_ATTEMPTS):
        sock = None
        try:
            sock = _esp32_open()
            for c in cmds:
                sock.sendall(c.encode())
            sock.settimeout(ESP32_POLL_TIMEOUT)      # drain phase: short polls, keep going until deadline
            data = b""
            deadline = time.time() + ESP32_BATCH_TIMEOUT
            while data.count(b"\n") < want and time.time() < deadline:
                try:
                    chunk = sock.recv(256)
                except socket.timeout:
                    continue                          # slow board: keep polling (no premature give-up)
                if not chunk:
                    break
                data += chunk
            replies = [ln.strip() for ln in data.decode(errors="ignore").splitlines() if ln.strip()]
            if replies:
                return replies
            last_err = "no response"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        finally:
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)   # make sure the ESP32 sees the close
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
        time.sleep(0.15 * (attempt + 1))              # brief backoff before a retry
    raise OSError(last_err)

async def send_esp32_batch(commands, quiet: bool = False):
    """Send commands over ONE serialized, short-lived connection.

    The global lock guarantees only a single connection to the controller exists at any
    moment (basic ESP32s wedge on concurrent clients). Returns replies ([] on failure).
    Set ``quiet=True`` to suppress logging (silent health check / manual test)."""
    if not state.esp32_ip:
        state.connection_status = "NO IP"
        return []
    cmds = [c.rstrip("\n") for c in commands]
    state.last_command = cmds[-1] if cmds else ""
    if not quiet:
        print(f"[esp32] >> {' | '.join(cmds)}", flush=True)
    global _esp32_last_ok
    async with _esp32_lock:
        try:
            responses = await asyncio.to_thread(_esp32_exchange_sync, cmds)
        except Exception as e:
            state.connection_status = "TCP ERROR"
            state.last_response = str(e)[:80]
            if not quiet:
                print(f"[esp32] << ERROR: {e}", flush=True)
            return []
    if responses:
        state.last_response = responses[-1]
        if not quiet:
            print(f"[esp32] << {' | '.join(responses)}", flush=True)
        if responses[-1] in ("OK", "PONG") or all(r == "OK" for r in responses):
            state.connection_status = f"CONNECTED ({state.esp32_ip})"
            _esp32_last_ok = time.time()
    else:
        state.connection_status = "TCP ERROR"
        state.last_response = "no response"
        if not quiet:
            print("[esp32] << (no response)", flush=True)
    return responses

async def esp32_health_loop():
    """Keep the link honest: if no command has gone through recently, poke the controller
    with PING. This confirms liveness and self-heals if a connection was ever dropped."""
    while True:
        await asyncio.sleep(ESP32_HEALTH_INTERVAL)
        if not state.esp32_ip:
            continue
        if time.time() - _esp32_last_ok < ESP32_HEALTH_INTERVAL:
            continue  # recently active -> already verified
        prev = state.connection_status
        await send_esp32_batch(["PING"], quiet=True)
        if state.connection_status != prev:
            print(f"[esp32] health check: {prev} -> {state.connection_status}", flush=True)

def queue_esp32_command(channel: int, op_state: int):
    """Enqueue a GPIO SET without touching the network (safe from any task/thread)."""
    _esp32_cmd_queue.put_nowait(f"SET {channel} {op_state}")

async def esp32_dispatch_loop():
    """Deliver queued GPIO commands off the inference path.

    Drains everything currently queued, coalesces to the LAST command per channel (so a
    backlog of stale SETs can never flood a weak board), then sends the batch over ONE
    short-lived connection -- still fully serialized through ``_esp32_lock``."""
    while True:
        cmd = await _esp32_cmd_queue.get()
        batch = [cmd]
        while not _esp32_cmd_queue.empty():
            try:
                batch.append(_esp32_cmd_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        latest = {}
        for c in batch:
            parts = c.split()
            if len(parts) >= 3 and parts[0] == "SET":
                latest[parts[1]] = c          # keep only the final state per channel
            else:
                latest[c] = c
        cmds = list(latest.values())
        if not cmds:
            continue
        try:
            await send_esp32_batch(cmds)
        except Exception as e:
            print(f"[esp32] dispatch error: {e}", flush=True)

async def send_esp32_command(channel: int, op_state: int, connect_timeout: float = 1.0):
    """Non-blocking GPIO command: queue it and return immediately.

    The FSM (which runs inline in the inference loop) must never wait on the ESP32 --
    a wedged/slow board would otherwise stall position updates for seconds. The
    dispatcher task performs the actual network exchange."""
    queue_esp32_command(channel, op_state)
    return True

# --- SPATIAL MATH & POLYGON FILTERING ---
def evaluate_point_against_polygons(pt_norm, frame_w, frame_h, cfg):
    """
    Evaluates if a normalized point (0-100) passes the inclusion polygon 
    and is outside all exclusion polygons using its bottom-center coordinate.
    """
    x_norm, y_norm = pt_norm
    inclusion = cfg.get("inclusion_polygon", [])
    exclusions = cfg.get("exclusion_polygons", [])
    
    pt_pixel = (x_norm * frame_w / 100.0, y_norm * frame_h / 100.0)

    # 1. Inclusion Polygon Check (Must be inside if polygon is defined)
    if inclusion and len(inclusion) >= 3:
        incl_pts = np.array([[(p[0] / 100.0) * frame_w, (p[1] / 100.0) * frame_h] for p in inclusion], dtype=np.float32)
        if cv2.pointPolygonTest(incl_pts, pt_pixel, False) < 0:
            return False

    # 2. Exclusion Polygons Check (Must be outside all exclusion zones)
    for excl in exclusions:
        if excl and len(excl) >= 3:
            excl_pts = np.array([[(p[0] / 100.0) * frame_w, (p[1] / 100.0) * frame_h] for p in excl], dtype=np.float32)
            if cv2.pointPolygonTest(excl_pts, pt_pixel, False) >= 0:
                return False

    return True

def project_to_baseline(box_bottom_center, p1, p2):
    x, y = box_bottom_center
    x1, y1 = p1
    x2, y2 = p2
    ba = np.array([x2 - x1, y2 - y1])
    pa = np.array([x - x1, y - y1])
    length_sq = np.dot(ba, ba)
    if length_sq == 0:
        return 0.0
    t = np.dot(pa, ba) / length_sq
    return float(t * 100.0)

# --- FORWARD OUTPUTS (independent of the threshold / reverse logic) ---
async def _eval_forward_outputs(positions: List[float], now: float):
    """Forward (non-reverse) outputs. They depend ONLY on Main + their own trigger
    and Max Activations. The threshold/reverse logic never touches these."""
    cfg_outputs = config["outputs"]
    for ch in range(2, 9):
        ch_str = str(ch)
        c_cfg = cfg_outputs[ch_str]
        if c_cfg["reverse"]:
            continue  # reverse outputs are handled by their own evaluator
        if state.output_states[ch_str] == 1:
            continue
        if c_cfg["depends"] and not state.main_active:
            continue
        if c_cfg["depends"] and state.output_activations_count[ch_str] >= c_cfg["max_act"]:
            continue
        trig = c_cfg["trigger"]
        # Gate: the person must be at/past the trigger but still within the nearness band.
        if any(trig <= p <= trig + ACTIVATION_BAND for p in positions):
            state.output_states[ch_str] = 1
            state.output_timers[ch_str] = now + c_cfg["duration"]
            state.output_activations_count[ch_str] += 1
            locked = ""
            if c_cfg["depends"] and state.output_activations_count[ch_str] >= c_cfg["max_act"]:
                locked = f" [max activations {c_cfg['max_act']} reached -> locked until Main completes]"
            print(f"[out {ch_str}] FIRED (forward, trigger {trig:g}, on for {c_cfg['duration']:g}s){locked}", flush=True)
            await send_esp32_command(ch, 1)

# --- REVERSE OUTPUTS (fully isolated; gated only by Main + the threshold) ---
async def _eval_reverse_outputs(cur_max, now: float):
    """Reverse outputs. They always depend on Main, arm only once everyone in frame is
    past the threshold point, and fire on a high->low crossing of their trigger."""
    cfg_outputs = config["outputs"]
    for ch in range(2, 9):
        ch_str = str(ch)
        c_cfg = cfg_outputs[ch_str]
        if not c_cfg["reverse"]:
            continue
        if state.output_states[ch_str] == 1:
            continue
        if not state.main_active or not state.reverse_armed:
            continue
        if state.output_activations_count[ch_str] >= c_cfg["max_act"]:
            continue
        trig = c_cfg["trigger"]
        if state.prev_max_position is None or cur_max is None:
            continue
        # Gate (mirrored): crossing from above to below AND still near the trigger point.
        if state.prev_max_position >= trig > cur_max and cur_max >= trig - ACTIVATION_BAND:
            state.output_states[ch_str] = 1
            state.output_timers[ch_str] = now + c_cfg["duration"]
            state.output_activations_count[ch_str] += 1
            print(f"[out {ch_str}] FIRED (reverse, trigger {trig:g}, on for {c_cfg['duration']:g}s)", flush=True)
            await send_esp32_command(ch, 1)

# --- FSM & AUTOMATION LOGIC ---
async def evaluate_fsm(positions: List[float]):
    global state
    cfg_outputs = config["outputs"]
    now = time.time()
    
    for ch_str in list(state.output_states.keys()):
        if state.output_states[ch_str] == 1 and now >= state.output_timers[ch_str]:
            state.output_states[ch_str] = 0
            await send_esp32_command(int(ch_str), 0)
            print(f"[out {ch_str}] OFF (duration elapsed)", flush=True)
            if ch_str == "1" or not cfg_outputs[ch_str].get("depends", False):
                # Any output that does not depend on Main (or Main itself) ends the
                # whole sequence when its duration expires -> global reset.
                await trigger_full_reset(f"Output {ch_str} duration expired")

    threshold = config["threshold_point"]
    threshold_delay = float(config.get("threshold_delay", 0.0) or 0.0)
    past_threshold_count = sum(1 for p in positions if p >= threshold)

    # Threshold crossing latch: recorded the first time anyone reaches the point.
    if past_threshold_count > 0 and state.threshold_crossed_at is None:
        state.threshold_crossed_at = now
        print(f"[threshold] CROSSED {threshold:g} (delay {threshold_delay:g}s)", flush=True)

    # Reverse logic stays inactive until the configured delay elapses after the cross.
    threshold_ready = (state.threshold_crossed_at is not None
                       and (now - state.threshold_crossed_at) >= threshold_delay)

    main_cfg = cfg_outputs["1"]
    main_trigger = main_cfg["trigger"]
    
    if state.output_states["1"] == 0:
        crossed_main = any(main_trigger <= p <= main_trigger + ACTIVATION_BAND for p in positions)
        if crossed_main and state.output_activations_count["1"] == 0:
            state.output_states["1"] = 1
            state.output_timers["1"] = now + main_cfg["duration"]
            state.output_activations_count["1"] = 1
            state.main_active = True
            print(f"[out 1] MAIN ON for {main_cfg['duration']:g}s (trigger {main_trigger:g})", flush=True)
            await send_esp32_command(1, 1)

    cur_max = max(positions) if positions else None

    # Reverse arming latch: only once Main is active, the threshold delay has elapsed,
    # AND everyone in frame is past the threshold point. Fully separate from forward outputs.
    if state.main_active and threshold_ready and positions and all(p >= threshold for p in positions):
        if not state.reverse_armed:
            print("[reverse] ARMED (Main active + threshold ready + everyone past threshold)", flush=True)
        state.reverse_armed = True

    await _eval_forward_outputs(positions, now)
    await _eval_reverse_outputs(cur_max, now)

    state.prev_max_position = cur_max

async def trigger_full_reset(reason: str, esp_budget: float = 3.0):
    state.threshold_crossed_at = None
    state.main_active = False
    state.reverse_armed = False
    state.prev_max_position = None
    for ch in range(1, 9):
        ch_str = str(ch)
        state.output_states[ch_str] = 0
        state.output_timers[ch_str] = 0.0
        state.output_activations_count[ch_str] = 0
    # Drive every GPIO LOW. Queued (non-blocking) -> the dispatcher coalesces these into
    # one connection. The FSM/inference loop must never wait on the controller network.
    print(f"[reset] {reason}", flush=True)
    for ch in range(1, 9):
        queue_esp32_command(ch, 0)

async def start_engine():
    """Full reset (all GPIO outputs LOW) BEFORE inference begins."""
    await trigger_full_reset("Engine start")
    reset_tracker()
    state.engine_running = True
    state.inference_fps = 0.0
    _start_capture()

async def _boot_sequence():
    """On boot: always drive all outputs LOW, then auto-start if configured."""
    await asyncio.sleep(1.5)  # let ESP32 discovery resolve the IP first
    await trigger_full_reset("Startup")
    if bool(config.get("auto_start", False)):
        state.engine_running = True
        state.inference_fps = 0.0
        _start_capture()

# --- PREPROCESSING / DECODING ---
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize so the largest side equals new_shape (aspect preserved), then pad."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def decode_person_boxes(out, conf_thres=0.45, iou_thres=0.45):
    """Decode raw YOLO logits [1, 4+nc, N] -> [[x1, y1, x2, y2, conf], ...] (class 0 = person).

    Person-only fast path: the class-0 score row is indexed directly instead of running an
    80-way argmax over every anchor each frame. That is a large per-frame CPU saving which
    directly lowers end-to-end latency (only class 0 = person is ever needed here).
    """
    raw = np.asarray(out[0])             # (4+nc, N)
    if raw.ndim != 2 or raw.shape[0] < 5:
        return []
    confs = raw[4]                       # class 0 = person
    mask = confs >= conf_thres
    if not np.any(mask):
        return []
    xywh = raw[:4].T[mask]               # (M, 4) -> center-x, center-y, w, h
    confs = confs[mask]
    xyxy = np.column_stack([
        xywh[:, 0] - xywh[:, 2] / 2.0,
        xywh[:, 1] - xywh[:, 3] / 2.0,
        xywh[:, 0] + xywh[:, 2] / 2.0,
        xywh[:, 1] + xywh[:, 3] / 2.0,
    ])
    keep = cv2.dnn.NMSBoxes(xyxy.tolist(), confs.tolist(), conf_thres, iou_thres)
    detections = []
    for i in np.array(keep, dtype=int).flatten():
        x1, y1, x2, y2 = xyxy[i]
        detections.append([float(x1), float(y1), float(x2), float(y2), float(confs[i])])
    return detections

# --- FRAME CAPTURE (single-slot buffer: newest frame wins, no pile-ups) ---
_frame_lock = threading.Lock()
_frame_event = threading.Event()
_latest_frame = None
_latest_frame_t = 0.0     # wall-clock time the newest frame was read (frame-age telemetry)
_frame_seq = 0
_stream_fps = 0.0
_capture_thread = None
_capture_enabled = False
_fps_count = 0
_fps_window_start = time.time()

def _capture_loop():
    """Continuously drain the RTSP stream, keeping ONLY the newest frame.

    Frames that arrive while inference is busy overwrite the single slot, so
    stale frames are dropped rather than piling up in a queue.
    """
    global _latest_frame, _latest_frame_t, _frame_seq, _stream_fps, _fps_count, _fps_window_start
    cap = None
    current_url = None
    announced = False
    while True:
        if not _capture_enabled:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.2)
            continue
        try:
            url = config["rtsp_url"]
            if cap is not None and url != current_url:
                cap.release()
                cap = None
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(url)
                current_url = url
                _fps_count = 0
                _fps_window_start = time.time()
                announced = False
                print(f"[capture] connecting to {url}", flush=True)
            ret, frame = cap.read()
            if not ret or frame is None:
                if cap is not None:
                    cap.release()
                    cap = None
                print("[capture] stream lost, reconnecting...", flush=True)
                time.sleep(0.2)
                continue
            if not announced:
                fh, fw = frame.shape[:2]
                print(f"[capture] streaming {fw}x{fh}", flush=True)
                announced = True
            now = time.time()
            with _frame_lock:
                _latest_frame = frame
                _latest_frame_t = now
                _frame_seq += 1
                _fps_count += 1
                window = now - _fps_window_start
                if window >= 2.0:
                    # Windowed average: immune to bursty reads under CPU load.
                    _stream_fps = round(_fps_count / window, 1)
                    state.stream_fps = _stream_fps
                    _fps_window_start = now
                    _fps_count = 0
            _frame_event.set()
        except Exception:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = None
            time.sleep(0.2)

def _start_capture():
    global _capture_thread, _capture_enabled, _fps_count, _fps_window_start
    _capture_enabled = True
    _fps_count = 0
    _fps_window_start = time.time()
    if _capture_thread is None or not _capture_thread.is_alive():
        _capture_thread = threading.Thread(target=_capture_loop, daemon=True, name="rtsp-capture")
        _capture_thread.start()

def _stop_capture():
    global _capture_enabled, _stream_fps
    _capture_enabled = False
    _stream_fps = 0.0
    state.stream_fps = 0.0

def _grab_snapshot_jpeg(quality=85, max_side=640):
    """Capture one frame (live capture if running, else a fresh connection) as JPEG bytes."""
    frame = None
    if _capture_enabled:
        with _frame_lock:
            if _latest_frame is not None:
                frame = _latest_frame.copy()
    if frame is None:
        cap = cv2.VideoCapture(config["rtsp_url"])
        try:
            for _ in range(40):
                ret, f = cap.read()
                if ret and f is not None:
                    frame = f
                    break
                time.sleep(0.1)
        finally:
            cap.release()
    if frame is None:
        return None
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest > max_side:  # scale so the largest side is max_side (per spec)
        s = max_side / float(longest)
        frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None

def _iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def _xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1e-6, x2 - x1), max(1e-6, y2 - y1)

def _cxcywh_to_xyxy(cx, cy, w, h):
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)

# --- Position output comes from the Kalman filter itself (no separate smoother) ---
def _kalman_noise_from_config():
    """Map the two user knobs to Kalman noise terms.

    ``position_smoothing`` (0..1) drives the measurement noise R: 0 -> the filter trusts the
    detector almost entirely (near-real-time, lowest latency; box jitter passes through),
    1 -> it trusts the motion model (steady dot, slightly laggy). ``track_agility`` (0..1)
    drives the velocity process noise Q: higher lets the estimated speed change quickly, so
    the reported position stays locked to sudden / fast movement.
    """
    s = config.get("position_smoothing", 0.25)
    if isinstance(s, bool):          # backward-compat with the old True/False setting
        s = 0.25 if s else 0.0
    try:
        s = min(max(float(s), 0.0), 1.0)
    except (TypeError, ValueError):
        s = 0.25
    a = config.get("track_agility", 0.5)
    try:
        a = min(max(float(a), 0.0), 1.0)
    except (TypeError, ValueError):
        a = 0.5
    r = 0.5 + (s * s) * 50.0           # 0 -> R=0.5 (raw) | 0.25 -> ~3.6 (old default) | 1 -> ~50
    q_vel = 500.0 + (a * a) * 30000.0  # 0.5 -> 8000 (old default)
    return 1.0, q_vel, r

# --- Constant-velocity Kalman track (state: cx, cy, w, h, vcx, vcy, vw, vh) ---
# The PREDICTED box is what we match against (so fast movers / brief dropouts still overlap the
# right track), and the FILTERED box is what we report -- the filter *is* the position smoother
# now. Tuning comes from `_kalman_noise_from_config()` and is live (no restart). No new deps.
class _Track:
    _H = np.zeros((4, 8), dtype=np.float64)
    for _i in range(4):
        _H[_i, _i] = 1.0

    def __init__(self, tid, box, conf, now, q_pos=1.0, q_vel=8000.0, r=4.0):
        cx, cy, w, h = _xyxy_to_cxcywh(box)
        self.id = tid
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([20.0, 20.0, 20.0, 20.0, 400.0, 400.0, 400.0, 400.0])
        self.now = now
        self.hits = 1
        self.conf = conf
        self.seen_high = True
        self.confirmed = False
        self.miss_ms = 0.0
        self.q_pos, self.q_vel, self.r = q_pos, q_vel, r

    def set_noise(self, q_pos, q_vel, r):
        """Live-update the filter tuning (config can change without a restart)."""
        self.q_pos, self.q_vel, self.r = q_pos, q_vel, r

    def predict(self, now):
        dt = min(max(now - self.now, 1e-3), 1.0)  # clamp huge gaps (stream stall) -> no blow-up
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        self.x = F @ self.x
        q = np.array([self.q_pos] * 4 + [self.q_vel] * 4) * dt
        self.P = F @ self.P @ F.T + np.diag(q)
        self.x[2] = max(1e-6, self.x[2])
        self.x[3] = max(1e-6, self.x[3])
        self.now = now
        self.miss_ms += dt * 1000.0

    def box(self):
        return _cxcywh_to_xyxy(self.x[0], self.x[1], self.x[2], self.x[3])

    def update(self, box, conf, high, now):
        cx, cy, w, h = _xyxy_to_cxcywh(box)
        z = np.array([cx, cy, w, h], dtype=np.float64)
        H = self._H
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.diag([self.r] * 4)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        self.hits += 1
        self.conf = conf
        self.seen_high = self.seen_high or high
        self.miss_ms = 0.0
        self.now = now

# Tracker state. Only ever touched by the single inference worker thread, EXCEPT
# reset_tracker() which start_engine() calls from the event loop -> guarded by _track_lock.
_tracks = []
_next_track_id = 1
_track_lock = threading.Lock()

def reset_tracker():
    global _tracks, _next_track_id
    with _track_lock:
        _tracks = []
        _next_track_id = 1

def _new_track(box, conf, now):
    global _next_track_id
    t = _Track(_next_track_id, box, conf, now, *_kalman_noise_from_config())
    _next_track_id += 1
    return t

def _confirm_tracks(dets, iou_thresh, confirm_hits, conf_high, sub_conf, validity_ms, now):
    """Associate detections to constant-velocity Kalman tracks; return the confirmed ones.

    Each track predicts its box forward to ``now`` *before* matching, so fast movers and
    brief dropouts still overlap the correct track. Association is greedy by DESCENDING IoU
    (best pairs claimed first), which stops two people crossing from swapping identities.

    Semantics preserved from the original design:
      * a detection >= ``conf_high`` seeds a track;
      * once seeded by a high-confidence hit, ANY detection >= ``sub_conf`` extends it;
      * a track registers after ``confirm_hits`` hits;
      * a track with no match keeps coasting (predicted) for ``validity_ms`` then is dropped.

    Only the single inference worker thread calls this (``reset_tracker`` is lock-guarded).
    """
    global _tracks
    with _track_lock:
        tracks = list(_tracks)

    # 1. Refresh the live filter tuning, predict every track's box to now, then build all
    #    viable (det, track) pairs against the PREDICTED boxes.
    noise = _kalman_noise_from_config()
    for t in tracks:
        t.set_noise(*noise)
        t.predict(now)

    pairs = []
    for di, d in enumerate(dets):
        conf = d[4]
        if conf < sub_conf:
            continue  # below both thresholds -> ignore entirely
        box = tuple(d[:4])
        for ti, t in enumerate(tracks):
            v = _iou(box, t.box())
            if v >= iou_thresh:
                pairs.append((v, di, ti))
    pairs.sort(key=lambda p: p[0], reverse=True)  # best overlaps first

    det_track = {}
    used_tracks = set()
    for v, di, ti in pairs:
        if di in det_track or ti in used_tracks:
            continue
        det_track[di] = ti
        used_tracks.add(ti)

    # 2. Update matched tracks / seed new ones.
    new_tracks = []
    confirmed = []
    for di, d in enumerate(dets):
        conf = d[4]
        is_high = conf >= conf_high
        ti = det_track.get(di)
        if ti is not None:
            t = tracks[ti]
            # A low-confidence detection can only extend a track already seeded by a
            # high-confidence hit; it may never start one.
            if not is_high and not t.seen_high:
                continue
            t.update(tuple(d[:4]), conf, is_high, now)
        elif is_high:
            t = _new_track(tuple(d[:4]), conf, now)
        else:
            continue  # low-conf with no seeded track to extend -> ignore
        t.confirmed = t.hits >= confirm_hits
        new_tracks.append(t)
        if t.confirmed:
            confirmed.append(t)

    # 3. Carry unmatched tracks while still inside the validity window (coasting).
    for ti, t in enumerate(tracks):
        if ti in used_tracks:
            continue
        if t.miss_ms <= validity_ms:
            new_tracks.append(t)
            if t.confirmed:
                confirmed.append(t)
        # else: dropped (identity released)

    with _track_lock:
        _tracks = new_tracks
    return confirmed

def _process_frame(frame, now=None):
    """Blocking preprocess -> infer -> decode -> track -> project (run in a worker thread).

    Returns ``(positions, tracks, infer_ms)`` where ``tracks`` is ``[{id, position}]`` in
    stable id order (so the UI feed no longer reshuffles between frames).
    """
    now = time.time() if now is None else now
    h_orig, w_orig = frame.shape[:2]
    inp_w, inp_h = state.model_input_size
    lb, r, (pad_x, pad_y) = letterbox(frame, (inp_h, inp_w))
    tensor = np.ascontiguousarray(
        cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    ).astype(np.float32) / 255.0

    # Snapshot the compiled model ONCE so a concurrent config-save reload can't swap it
    # out mid-frame (that used to raise a KeyError on the output-port lookup). Index the
    # result by position [0], which is independent of which model instance produced it.
    cm = ov_compiled_model
    if cm is None:
        return [], [], 0.0
    infer_start = time.time()
    res = cm([tensor])[0]
    infer_ms = (time.time() - infer_start) * 1000.0

    conf_thres = float(config.get("conf_threshold", 0.70) or 0.70)
    iou_match = float(config.get("iou_match_threshold", 0.30) or 0.30)
    confirm_hits = int(config.get("min_consecutive_frames", 3) or 1)
    sub_conf = float(config.get("subsequent_conf_threshold", 0.50) or 0.0)
    validity_ms = float(config.get("track_validity_ms", 500) or 0.0)

    # Decode down to the LOWER of the two thresholds so low-confidence detections are
    # still available to extend already-confirmed tracks (see _confirm_tracks).
    decode_thres = min(conf_thres, sub_conf) if sub_conf > 0 else conf_thres
    dets = decode_person_boxes(res, conf_thres=decode_thres, iou_thres=0.45)

    # Spatial gating FIRST, using each box's BOTTOM-CENTER point only.
    # A detection is discarded entirely (never tracked, never fed to the FSM or the
    # status feed) if its bottom-center is outside the inclusion polygon or inside
    # any exclusion polygon.
    p1 = config["base_line"]["p1"]
    p2 = config["base_line"]["p2"]
    candidates = []
    for x1, y1, x2, y2, conf in dets:
        ox1 = (x1 - pad_x) / r
        ox2 = (x2 - pad_x) / r
        oy2 = (y2 - pad_y) / r
        cx = float(np.clip((ox1 + ox2) / 2.0, 0.0, w_orig - 1.0))   # bottom-center X
        by = float(np.clip(oy2, 0.0, h_orig - 1.0))                 # bottom-center Y
        bottom_center_norm = (cx / w_orig * 100.0, by / h_orig * 100.0)
        if evaluate_point_against_polygons(bottom_center_norm, w_orig, h_orig, config):
            candidates.append([x1, y1, x2, y2, conf, bottom_center_norm])

    # Kalman association + temporal confirmation. The reported position IS the Kalman-filtered
    # box (the filter doubles as the smoother) -- tune it via position_smoothing/track_agility.
    confirmed = _confirm_tracks(candidates, iou_match, confirm_hits, conf_thres, sub_conf, validity_ms, now)

    tracks = []
    for t in confirmed:
        # Progress = perpendicular projection of the FILTERED bottom-center onto the base line.
        x1, y1, x2, y2 = t.box()
        cx = float(np.clip(((x1 + x2) / 2.0 - pad_x) / r, 0.0, w_orig - 1.0))
        by = float(np.clip((y2 - pad_y) / r, 0.0, h_orig - 1.0))
        pos = project_to_baseline((cx / w_orig * 100.0, by / h_orig * 100.0), p1, p2)
        tracks.append({"id": t.id, "position": round(float(pos), 1)})
    tracks.sort(key=lambda d: d["id"])          # stable order (no per-frame reshuffle)
    tracks = tracks[:10]
    return [d["position"] for d in tracks], tracks, infer_ms

def _wait_and_process(last_seq, timeout=0.5):
    """One thread hop: wait for the newest frame, then run the whole pipeline on it.

    Merging the frame hand-off and processing into a single ``to_thread`` call removes an
    extra thread round-trip per frame. Stale frames are still dropped: inference only ever
    sees the newest frame captured (single-slot buffer)."""
    _frame_event.wait(timeout)
    _frame_event.clear()
    with _frame_lock:
        frame = _latest_frame
        seq = _frame_seq
        frame_t = _latest_frame_t
    if frame is None or seq == last_seq:
        return None
    now = time.time()
    age_ms = max(0.0, (now - frame_t) * 1000.0) if frame_t else 0.0
    positions, tracks, infer_ms = _process_frame(frame, now)
    return seq, positions, tracks, infer_ms, age_ms

async def inference_worker():
    global ov_compiled_model
    simulate = os.environ.get("HALO_SIMULATE", "0") == "1"
    last_seq = -1
    last_t = None
    while True:
        # Idle until the engine is started from the UI (or auto_start is enabled).
        if not state.engine_running:
            state.detected_persons = []
            state.tracks = []
            state.inference_latency_ms = 0.0
            state.last_inference_ms = 0.0
            state.inference_fps = 0.0
            state.frame_age_ms = 0.0
            state.frame_lag = 0.0
            state.frames_skipped = 0
            last_seq = -1          # restart the frame-skip counter on the next engine run
            await asyncio.sleep(0.3)
            continue

        loop_start = time.time()
        try:
            if ov_compiled_model is None:
                state.inference_latency_ms = 0.0
                state.last_inference_ms = 0.0
                state.detected_persons = [round(float(np.random.uniform(0, 100)), 1)] if simulate else []
                state.tracks = [{"id": 1, "position": state.detected_persons[0]}] if state.detected_persons else []
                await asyncio.sleep(0.3)
                continue

            # One thread hop: wait for the newest captured frame, then process it. If
            # inference is still busy when a newer frame arrives, the slot has already been
            # overwritten -> the stale frame is dropped (no pile-up / no growing latency).
            result = await asyncio.to_thread(_wait_and_process, last_seq)
            if result is None:
                await asyncio.sleep(0.002)
                continue
            seq, positions, tracks, infer_ms, age_ms = result
            # Frames the capture thread replaced while we were busy -- proves we always take
            # the newest frame (a growing count here would mean we're falling behind).
            if last_seq >= 0:
                state.frames_skipped += max(0, seq - last_seq - 1)
            last_seq = seq
            state.frame_age_ms = round(age_ms, 1)
            _sfps = state.stream_fps
            state.frame_lag = round(age_ms / (1000.0 / _sfps), 2) if _sfps > 0 else 0.0

            state.inference_latency_ms = round(infer_ms, 2)
            state.last_inference_ms = round(infer_ms, 2)
            state.inference_ms_avg = round(
                infer_ms if state.inference_ms_avg == 0.0 else state.inference_ms_avg * 0.8 + infer_ms * 0.2,
                2,
            )

            state.detected_persons = positions
            state.tracks = tracks
            # evaluate_fsm only enqueues ESP32 commands now, so it never blocks this loop.
            await evaluate_fsm(state.detected_persons)

        except Exception as e:
            print(f"[inference] error: {e}", flush=True)
            state.inference_latency_ms = 0.0
            await asyncio.sleep(0.5)

        # Rate-limit inference to the configurable cap. Skip the wait entirely when a newer
        # frame is already waiting -> never add latency we don't have to.
        max_fps = float(config.get("max_fps", 10) or 10)
        if max_fps > 0:
            with _frame_lock:
                pending = _frame_seq != last_seq
            if not pending:
                remaining = (1.0 / max_fps) - (time.time() - loop_start)
                if remaining > 0:
                    await asyncio.sleep(remaining)

        now = time.time()
        if last_t is not None and now > last_t:
            inst = 1.0 / (now - last_t)
            state.inference_fps = round(inst if state.inference_fps == 0.0 else state.inference_fps * 0.8 + inst * 0.2, 1)
        last_t = now

@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(initialize_openvino_model)
    asyncio.create_task(esp32_discovery_loop())
    asyncio.create_task(esp32_health_loop())
    asyncio.create_task(esp32_dispatch_loop())
    asyncio.create_task(inference_worker())
    asyncio.create_task(_boot_sequence())

# --- API ENDPOINTS ---
def _status_payload():
    """Single source of truth for the status payload (used by /api/status and SSE)."""
    return {
        "connection_status": state.connection_status,
        "esp32_ip": state.esp32_ip,
        "last_command": state.last_command,
        "last_response": state.last_response,
        "detected_persons": state.detected_persons,
        "tracks": state.tracks,
        "inference_latency_ms": state.inference_latency_ms,
        "output_states": state.output_states,
        "output_activations_count": state.output_activations_count,
        "main_active": state.main_active,
        "model_loaded": state.model_loaded,
        "model_device": state.model_device,
        "model_input_size": state.model_input_size,
        "last_inference_ms": state.last_inference_ms,
        "inference_ms_avg": state.inference_ms_avg,
        "inference_fps": state.inference_fps,
        "stream_fps": state.stream_fps,
        "frame_age_ms": state.frame_age_ms,
        "frame_lag": state.frame_lag,
        "frames_skipped": state.frames_skipped,
        "engine_running": state.engine_running,
        "device_warning": state.device_warning,
        "esp32_host": config.get("esp32_host", ""),
        "esp32_tcp_port": config.get("esp32_tcp_port", 5000),
        "base_line": config.get("base_line", {"p1": [0, 0], "p2": [100, 0]}),
        "threshold_point": config.get("threshold_point", 50.0),
        "outputs": config.get("outputs", {}),
    }

@app.get("/api/status")
async def get_status():
    return _status_payload()

@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events push: streams the status payload at ~10 Hz over one persistent
    connection so the UI no longer pays the ~300 ms polling latency floor."""
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = json.dumps(_status_payload())
            except Exception:
                payload = "{}"
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.post("/api/control")
async def control_action(payload: dict):
    action = payload.get("action")
    if action == "off":
        await trigger_full_reset("Manual OFF command")
        return {"status": "ok"}
    elif action == "reset":
        await trigger_full_reset("Manual RESET command")
        return {"status": "ok"}
    elif action == "start":
        await start_engine()
        return {"status": "ok"}
    elif action == "stop":
        state.engine_running = False
        _stop_capture()
        await trigger_full_reset("Engine stopped")
        return {"status": "ok"}
    elif action == "trigger":
        ch = int(payload.get("channel", 1))
        state.output_states[str(ch)] = 1
        state.output_timers[str(ch)] = time.time() + config["outputs"][str(ch)]["duration"]
        if ch == 1:
            state.main_active = True
        await send_esp32_command(ch, 1)
        return {"status": "ok"}
    raise HTTPException(400, "Invalid action")

@app.get("/api/config", response_class=JSONResponse)
async def get_config():
    return config

@app.post("/api/config")
async def update_config(new_config: dict):
    global config
    # Reverse outputs ALWAYS depend on Main, regardless of what the UI sent.
    for o in (new_config.get("outputs") or {}).values():
        if isinstance(o, dict) and o.get("reverse"):
            o["depends"] = True
    prev_device = config.get("device_priority")
    config = new_config
    save_config(config)
    # Only recompile when the accelerator preference changed (or the model never loaded).
    # Every other setting is read live by the inference loop, so recompiling here would only
    # stall inference -- and used to race the inference thread (KeyError on the output port).
    if config.get("device_priority") != prev_device or not state.model_loaded:
        await asyncio.to_thread(initialize_openvino_model)
    return {"status": "saved"}

@app.get("/api/snapshot")
async def get_snapshot():
    jpg = await asyncio.to_thread(_grab_snapshot_jpeg)
    if jpg is None:
        raise HTTPException(503, "Could not capture a snapshot from the stream")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

@app.get("/api/esp32/test")
async def esp32_test():
    ip = (state.esp32_ip or (config.get("esp32_host") or "")).strip()
    if not ip:
        return {"reachable": False, "ip": None, "error": "no ESP32 IP configured or discovered"}
    state.esp32_ip = ip
    t0 = time.time()
    # Route the test through the SAME serialized single-connection path as everything else.
    replies = await send_esp32_batch(["PING"], quiet=True)
    ms = round((time.time() - t0) * 1000.0, 1)
    ok = bool(replies) and replies[-1] == "PONG"
    return {"reachable": ok, "ip": ip, "port": int(config.get("esp32_tcp_port", 5000)), "ms": ms,
            "error": "" if ok else (state.last_response or "no response")}

@app.post("/api/geometry")
async def update_geometry(payload: dict):
    """Save just the calibration geometry (no model reload)."""
    if "inclusion_polygon" in payload:
        config["inclusion_polygon"] = payload["inclusion_polygon"]
    if "exclusion_polygons" in payload:
        config["exclusion_polygons"] = payload["exclusion_polygons"]
    if "base_line" in payload:
        config["base_line"] = payload["base_line"]
    if "threshold_point" in payload:
        config["threshold_point"] = payload["threshold_point"]
    save_config(config)
    return {"status": "saved"}

# --- HTML UI (`/execute` and `/setup`) ---
EXECUTE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Halloween Automation - Execute</title>
  <style>
    body { font-family: -apple-system, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; }
    h2 { color: #38bdf8; text-align: center; }
    .card { background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .status-bar { display: flex; justify-content: space-between; font-size: 14px; background: #334155; padding: 10px; border-radius: 8px; margin-bottom: 16px; }
    .latency-badge { background: #0f172a; color: #38bdf8; padding: 4px 8px; border-radius: 6px; font-family: monospace; font-weight: bold; }
    .outputs-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .output-btn { padding: 12px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; color: white; background: #475569; }
    .output-btn.active { background: #16a34a; box-shadow: 0 0 10px #16a34a; }
    .global-btns { display: flex; gap: 10px; margin-top: 16px; }
    .btn-off { background: #dc2626; flex: 1; padding: 12px; border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; }
    .btn-reset { background: #ca8a04; flex: 1; padding: 12px; border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; }
    nav { text-align: center; margin-bottom: 20px; }
    nav a { color: #38bdf8; text-decoration: none; font-weight: bold; }
  </style>
</head>
<body>
  <div class="container">
    <nav><a href="/setup">-> Go to Setup Configuration</a></nav>
    <h2>Halloween Execution Dashboard</h2>
    
    <div class="status-bar" id="statusBar">Loading connection status...</div>

    <div id="deviceWarning" style="display:none; background:#7f1d1d; color:#fecaca; border:3px solid #dc2626; border-radius:8px; padding:14px; margin-bottom:12px; font-family:monospace; font-weight:bold; text-align:center; white-space:pre-line;"></div>

    <div class="card" style="display: flex; justify-content: space-between; align-items: center;">
      <div>
        <h3 style="margin: 0 0 4px 0;">Engine</h3>
        <span style="font-size: 12px; color: #94a3b8;" id="engineState">STOPPED</span>
      </div>
      <button id="engineBtn" class="output-btn" style="min-width: 150px; background: #16a34a;" onclick="toggleEngine()">START ENGINE</button>
    </div>

    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <h3 style="margin: 0 0 4px 0;">ESP32 GPIO Controller</h3>
          <span id="esp32State" style="font-size: 12px; color: #94a3b8;">...</span>
        </div>
        <button class="output-btn" style="min-width: 90px;" onclick="testEsp32()">TEST</button>
      </div>
      <div style="font-size: 12px; color: #94a3b8; margin-top: 10px; font-family: monospace; line-height: 1.6;">
        <div>IP: <span id="esp32Ip">-</span></div>
        <div>Last cmd: <span id="esp32Cmd">-</span></div>
        <div>Last resp: <span id="esp32Resp">-</span></div>
      </div>
    </div>

    <div class="card" style="display: flex; justify-content: space-between; align-items: center;">
      <div>
        <h3 style="margin: 0 0 4px 0;">OpenVINO Inference Time</h3>
        <span style="font-size: 12px; color: #94a3b8;" id="modelInfo">YOLO26m (640x640, FP16)</span>
      </div>
      <div class="latency-badge" id="latencyDisplay">0.00 ms</div>
    </div>

    <div class="card">
      <h3>Linear Model Visualization (Up to 10)</h3>
      <canvas id="lineViz" width="560" height="96" style="width:100%; height:auto; background:#0f172a; border:1px solid #334155; border-radius:8px;"></canvas>
      <p id="positionsFeed" style="font-family: monospace; font-size: 18px; color: #38bdf8; margin:10px 0 0 0;">--</p>
    </div>

    <div class="card">
      <h3>Outputs Control Panel</h3>
      <div class="outputs-grid" id="outputsGrid"></div>
      <div class="global-btns">
        <button class="btn-off" onclick="sendControl('off')">GLOBAL OFF</button>
        <button class="btn-reset" onclick="sendControl('reset')">FULL RESET</button>
      </div>
    </div>
  </div>

  <script>
    let outputNames = {};
    let engineRunning = false;
    async function toggleEngine() {
      const action = engineRunning ? 'stop' : 'start';
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: action}) });
      updateStatus();
    }
    async function testEsp32() {
      const el = document.getElementById('esp32State');
      if (el) { el.textContent = 'TESTING...'; el.style.color = '#38bdf8'; }
      try {
        const d = await (await fetch('/api/esp32/test')).json();
        if (el) {
          el.textContent = d.reachable ? `CONNECTED (${d.ip}) ${d.ms} ms` : `FAILED: ${d.error || 'unreachable'}`;
          el.style.color = d.reachable ? '#22c55e' : '#ef4444';
        }
      } catch (e) { if (el) { el.textContent = 'TEST ERROR'; el.style.color = '#ef4444'; } }
      updateStatus();
    }
    async function loadNames() {
      const res = await fetch('/api/config');
      const cfg = await res.json();
      outputNames = cfg.output_names;
    }
    // Draw the linear-model visualization: baseline, output triggers, threshold, and a
    // labeled dot per tracked person (stable id => the dot you watch stays the same).
    function drawLineViz(data) {
      const cv = document.getElementById('lineViz');
      if (!cv) return;
      const ctx = cv.getContext('2d');
      const W = cv.width, H = cv.height, pad = 34;
      ctx.clearRect(0, 0, W, H);
      const xOf = (p) => pad + (Math.max(0, Math.min(100, p)) / 100) * (W - 2 * pad);
      const axisY = 40;
      ctx.strokeStyle = '#475569'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(pad, axisY); ctx.lineTo(W - pad, axisY); ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
      for (let p = 0; p <= 100; p += 25) {
        const x = xOf(p);
        ctx.beginPath(); ctx.moveTo(x, axisY - 4); ctx.lineTo(x, axisY + 4); ctx.stroke();
        ctx.fillText(p.toString(), x, axisY + 20);
      }
      const outs = data.outputs || {};
      for (let i = 1; i <= 8; i++) {
        const o = outs[i.toString()];
        if (!o) continue;
        const x = xOf(o.trigger);
        const active = (data.output_states || {})[i.toString()] === 1;
        ctx.strokeStyle = active ? '#16a34a' : (i === 1 ? '#f59e0b' : '#334155');
        ctx.lineWidth = active ? 3 : 1;
        ctx.beginPath(); ctx.moveTo(x, axisY - 22); ctx.lineTo(x, axisY + 8); ctx.stroke();
        ctx.fillStyle = active ? '#22c55e' : '#94a3b8';
        ctx.fillText(i.toString(), x, axisY - 26);
      }
      if (typeof data.threshold_point === 'number') {
        const x = xOf(data.threshold_point);
        ctx.strokeStyle = '#eab308'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, axisY - 22); ctx.lineTo(x, axisY + 8); ctx.stroke();
        ctx.setLineDash([]);
      }
      const tracks = data.tracks || [];
      tracks.forEach((t, idx) => {
        const x = xOf(t.position);
        const y = axisY + 30 + (idx % 2) * 12;
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#e0f2fe';
        ctx.fillText('#' + t.id + ' ' + t.position.toFixed(1), x, y + 14);
      });
    }

    // Apply a status payload (used by both SSE push and the polling fallback).
    function applyStatus(data) {
      document.getElementById('statusBar').innerHTML = `
          <span><b>Status:</b> ${data.connection_status}</span>
          <span><b>Last Resp:</b> ${data.last_response}</span>
        `;
        const infMs = data.inference_latency_ms || 0;
        const infAvg = data.inference_ms_avg || 0;
        document.getElementById('latencyDisplay').innerText = `${infMs.toFixed(2)} ms`;
        const fps = data.inference_fps || 0;
        const sfps = data.stream_fps || 0;
        const lag = data.frame_lag || 0;
        const age = data.frame_age_ms || 0;
        const skipped = data.frames_skipped || 0;
        document.getElementById('modelInfo').innerText =
          `${data.model_loaded ? 'YOLO26m' : 'Model not loaded'} | ${data.model_device} | ` +
          `${(data.model_input_size || [640, 640]).join('x')} | avg ${infAvg.toFixed(1)} ms | ` +
          `${fps.toFixed(1)} FPS (stream ${sfps.toFixed(1)}) | lag ${lag.toFixed(1)}f (${age.toFixed(0)}ms) | skipped ${skipped}`;
        const tracks = data.tracks || [];
        document.getElementById('positionsFeed').innerText =
          tracks.length > 0 ? tracks.map(t => `#${t.id}: ${t.position.toFixed(1)}`).join('   ') : "No persons detected";
        drawLineViz(data);

        engineRunning = !!data.engine_running;
        const eb = document.getElementById('engineBtn');
        if (eb) {
          eb.textContent = engineRunning ? 'STOP ENGINE' : 'START ENGINE';
          eb.style.background = engineRunning ? '#dc2626' : '#16a34a';
          eb.style.boxShadow = engineRunning ? '0 0 10px #dc2626' : 'none';
        }
        const es = document.getElementById('engineState');
        if (es) es.textContent = engineRunning ? 'RUNNING' : 'STOPPED';

        const dw = document.getElementById('deviceWarning');
        if (dw) {
          if (data.device_warning) { dw.style.display = 'block'; dw.textContent = data.device_warning; }
          else { dw.style.display = 'none'; }
        }

        const es32 = document.getElementById('esp32State');
        if (es32) {
          const s = data.connection_status || 'UNKNOWN';
          es32.textContent = s;
          es32.style.color = s.includes('CONNECTED') ? '#22c55e' : (s.includes('SEARCH') ? '#eab308' : '#ef4444');
        }
        const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        setTxt('esp32Ip', data.esp32_ip || data.esp32_host || '-');
        setTxt('esp32Cmd', data.last_command || '-');
        setTxt('esp32Resp', data.last_response || '-');

        let gridHtml = '';
        for (let i = 1; i <= 8; i++) {
          const chStr = i.toString();
          const isActive = data.output_states[chStr] === 1;
          const name = (i === 1) ? "Main (Output 1)" : (outputNames[`out_${i}`] || `Output ${i}`);
          gridHtml += `<button class="output-btn ${isActive ? 'active' : ''}" onclick="triggerOutput(${i})">${name}</button>`;
        }
        document.getElementById('outputsGrid').innerHTML = gridHtml;
    }
    async function updateStatus() {
      try { applyStatus(await (await fetch('/api/status')).json()); }
      catch (e) { console.error(e); }
    }
    // Prefer SSE push (lower latency, one persistent connection). Fall back to 300 ms
    // polling only if EventSource can't be established or drops.
    function connectEvents() {
      try {
        const src = new EventSource('/api/events');
        src.onmessage = (e) => { try { applyStatus(JSON.parse(e.data)); } catch (_) {} };
        src.onerror = () => { src.close(); setTimeout(connectEvents, 2000); };
      } catch (_) {
        setInterval(updateStatus, 300);
      }
    }
    async function sendControl(action) {
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: action}) });
      updateStatus();
    }
    async function triggerOutput(ch) {
      await fetch('/api/control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'trigger', channel: ch}) });
      updateStatus();
    }
    loadNames();
    updateStatus();
    connectEvents();
  </script>
</body>
</html>
"""

SETUP_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Halloween Automation - Setup</title>
  <style>
    body { font-family: -apple-system, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; }
    h2 { color: #38bdf8; text-align: center; }
    .card { background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    label { display: block; margin-top: 8px; font-size: 14px; color: #94a3b8; }
    input, select { width: 100%; padding: 8px; margin-top: 4px; background: #334155; border: 1px solid #475569; color: white; border-radius: 6px; }
    .hint { font-size: 12px; color: #94a3b8; line-height: 1.45; margin: 4px 0 10px 0; }
    .hint b { color: #cbd5e1; }
    button.save { background: #16a34a; color: white; border: none; padding: 12px; width: 100%; border-radius: 8px; font-weight: bold; cursor: pointer; margin-top: 16px; }
    button.tool { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 8px 10px; font-weight: bold; cursor: pointer; font-size: 13px; }
    nav { text-align: center; margin-bottom: 20px; }
    nav a { color: #38bdf8; text-decoration: none; font-weight: bold; }
  </style>
</head>
<body>
  <div class="container">
    <nav><a href="/execute"><- Back to Execution Dashboard</a></nav>
    <h2>Setup & Calibration</h2>

    <div class="card">
      <h3>System Configuration</h3>
      <label>Device Priority (accelerator order)</label>
      <select id="device_priority">
        <option value="NPU,GPU,CPU">NPU → GPU → CPU (recommended)</option>
        <option value="GPU,NPU,CPU">GPU → NPU → CPU</option>
        <option value="NPU,CPU">NPU → CPU</option>
        <option value="GPU,CPU">GPU → CPU</option>
        <option value="CPU">CPU only</option>
      </select>
      <label>RTSP Stream URL</label>
      <input type="text" id="rtsp_url" placeholder="rtsp://user:pass@192.168.1.x:554/stream">
      <label>ESP32 GPIO Controller IP (blank = auto-discover)</label>
      <input type="text" id="esp32_host" placeholder="192.168.1.x">
      <label>Threshold Point (0-100)</label>
      <input type="number" id="threshold_point" step="0.1">
      <label>Threshold Delay (seconds after threshold cross)</label>
      <input type="number" id="threshold_delay" step="0.1" min="0">
      <label>Max Inference FPS (cap)</label>
      <input type="number" id="max_fps" step="1" min="1">
      <div style="font-size:12px; color:#94a3b8; margin:-4px 0 10px 0;">
        Note: higher = lower latency (fresher positions). Set as high as the accelerator can
        sustain &mdash; stale frames are dropped automatically, so a higher cap never adds lag.
      </div>
      <label style="margin-top: 12px;"><input type="checkbox" id="auto_start" style="width: auto; margin-right: 6px;"> Auto-start engine on boot</label>
    </div>

    <div class="card">
      <h3>Detection Filtering &amp; Tracking</h3>
      <label>Person Confidence Threshold (0-1)</label>
      <input type="number" id="conf_threshold" step="0.01" min="0.1" max="0.99">
      <label>Min Box Overlap / IoU to Track Same Person (0-1)</label>
      <input type="number" id="iou_match_threshold" step="0.01" min="0.05" max="0.9">
      <label>Consecutive Detections Required to Register</label>
      <input type="number" id="min_consecutive_frames" step="1" min="1" max="30">
      <label>Subsequent Confidence Threshold (0-1, for already-tracked persons)</label>
      <input type="number" id="subsequent_conf_threshold" step="0.01" min="0" max="0.99">
      <label>Track Validity Window (ms)</label>
      <input type="number" id="track_validity_ms" step="50" min="0" max="5000" placeholder="500">
      <div class="hint">Suggested <b>300&ndash;800 ms</b> (default 500). How long a person may be missed
        before their track is dropped. Shorter = reacts faster when people leave the frame; longer =
        rides out brief detector dropouts without losing the track.</div>

      <div style="border-top:1px solid #334155; margin-top:14px; padding-top:12px;">
        <label style="margin-top:0;">Position Filter (Kalman)</label>
        <div class="hint" style="margin-top:6px;">The reported 0&ndash;100 position is the <b>Kalman-filtered</b> box bottom-center
          (feeding the display, triggers and baseline projection). It does <b>not</b> affect what the detector
          sees. These two knobs trade latency against steadiness &mdash; no restart needed.</div>
        <label>Position Smoothing &mdash; 0 to 1</label>
        <input type="number" id="position_smoothing" step="0.05" min="0" max="1" placeholder="0.25">
        <div class="hint"><b>0 = near-real-time</b> (lowest latency, raw-ish &mdash; some jitter). <b>1 = heavy</b>
          smoothing (steady dot, slightly laggy). Suggested 0&ndash;0.4 for a jump scare; default 0.25.</div>
        <label>Track Agility &mdash; 0 to 1</label>
        <input type="number" id="track_agility" step="0.05" min="0" max="1" placeholder="0.50">
        <div class="hint">How quickly the filter follows <b>sudden changes in speed</b>. Higher = the reported
          position stays locked to fast/lunging movement; lower = steadier but slower to react. Suggested
          0.3&ndash;0.8; default 0.50.</div>
        <div class="hint" style="color:#7c8899;">Real-time start: <b>0.25 / 0.50</b>. Want it snappier &rarr; lower
          Position Smoothing. Dot lags a runner &rarr; raise Track Agility. Dot jitters when still &rarr; raise Smoothing.</div>
      </div>
    </div>

    <div class="card">
      <h3>Spatial Geometry &amp; Calibration</h3>
      <div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px;">
        <button type="button" class="tool" onclick="grabSnapshot()">Capture Snapshot</button>
        <button type="button" class="tool" id="btnIncl" onclick="setMode('inclusion')">Inclusion (4 pts)</button>
        <button type="button" class="tool" id="btnExcl" onclick="setMode('exclusion')">Exclusion (4 pts)</button>
        <button type="button" class="tool" id="btnLine" onclick="setMode('line')">Base Line (2 pts)</button>
        <button type="button" class="tool" onclick="undoLast()">Undo</button>
        <button type="button" class="tool" onclick="clearAllGeo()">Clear All</button>
      </div>
      <div id="snapWrap" style="position:relative; background:#0f172a; border:1px solid #334155; border-radius:8px; overflow:hidden;">
        <img id="snapImg" style="display:none; width:100%; height:auto; vertical-align:top;" alt="snapshot" />
        <canvas id="snapCanvas" style="position:absolute; left:0; top:0; width:100%; height:100%; display:none; cursor:crosshair;"></canvas>
        <div id="snapHint" style="color:#64748b; font-size:14px; padding:40px; text-align:center;">Capture a snapshot to begin drawing</div>
      </div>
      <div style="font-size:12px; color:#94a3b8; margin-top:8px;">
        <span style="color:#22c55e;">&#9632;</span> Inclusion &nbsp;
        <span style="color:#ef4444;">&#9632;</span> Exclusion &nbsp;
        <span style="color:#38bdf8;">&#9632;</span> Base line &nbsp;
        <span style="color:#eab308;">&#9632;</span> Threshold
      </div>
      <div id="geoStatus" style="font-size:12px; color:#38bdf8; margin-top:6px; font-family:monospace;"></div>
    </div>

    <div class="card">
      <h3>Output 1 (Main)</h3>
      <label>Trigger Position (0-100)</label>
      <input type="number" id="out1_trigger" step="0.1">
      <label>Duration (Seconds)</label>
      <input type="number" id="out1_duration" step="0.1" min="0.1">
    </div>

    <div class="card">
      <h3>Outputs 2-8 Configuration</h3>
      <div id="outputsConfig"></div>
    </div>

    <button class="save" onclick="saveConfig()">Save Configuration</button>
  </div>

  <script>
    // ---- Spatial geometry & calibration drawing ----
    let geo = { inclusion: [], exclusions: [], line: { p1: null, p2: null } };
    let draft = [];
    let mode = null;
    const snapImg = document.getElementById('snapImg');
    const snapCanvas = document.getElementById('snapCanvas');
    const snapCtx = snapCanvas.getContext('2d');

    function setMode(m) { mode = (mode === m) ? null : m; draft = []; refreshTools(); redraw(); }
    function refreshTools() {
      const map = { inclusion: 'btnIncl', exclusion: 'btnExcl', line: 'btnLine' };
      for (const m in map) { const el = document.getElementById(map[m]); if (el) el.style.outline = (mode === m) ? '2px solid #38bdf8' : 'none'; }
      const hints = {
        inclusion: 'Click 4 points to define the inclusion polygon',
        exclusion: 'Click 4 points to define an exclusion zone',
        line: 'Click 2 points to define the base line (low -> high)'
      };
      const hint = document.getElementById('snapHint');
      if (snapImg.style.display === 'none') hint.textContent = 'Capture a snapshot to begin drawing';
      else hint.textContent = mode ? hints[mode] : 'Pick a tool, then click on the image';
    }
    function px(pt) { return { x: pt[0] / 100 * snapCanvas.width, y: pt[1] / 100 * snapCanvas.height }; }
    function drawPoly(pts, stroke, fill) {
      if (!pts || pts.length < 2) return;
      snapCtx.beginPath();
      pts.forEach((pt, i) => { const p = px(pt); i ? snapCtx.lineTo(p.x, p.y) : snapCtx.moveTo(p.x, p.y); });
      if (pts.length >= 3) snapCtx.closePath();
      snapCtx.fillStyle = fill; snapCtx.strokeStyle = stroke; snapCtx.lineWidth = 2;
      if (pts.length >= 3) snapCtx.fill();
      snapCtx.stroke();
    }
    function dot(p, color, r) {
      snapCtx.beginPath(); snapCtx.arc(p.x, p.y, r || 4, 0, Math.PI * 2); snapCtx.fillStyle = color; snapCtx.fill();
    }
    function redraw() {
      if (!snapCanvas.width) return;
      snapCtx.clearRect(0, 0, snapCanvas.width, snapCanvas.height);
      drawPoly(geo.inclusion, '#22c55e', 'rgba(34,197,94,0.15)');
      geo.exclusions.forEach(p => drawPoly(p, '#ef4444', 'rgba(239,68,68,0.15)'));
      if (geo.line.p1 && geo.line.p2) {
        const a = px(geo.line.p1), b = px(geo.line.p2);
        snapCtx.beginPath(); snapCtx.moveTo(a.x, a.y); snapCtx.lineTo(b.x, b.y);
        snapCtx.strokeStyle = '#38bdf8'; snapCtx.lineWidth = 2; snapCtx.stroke();
        dot(a, '#38bdf8'); dot(b, '#38bdf8');
        const t = (parseFloat(document.getElementById('threshold_point').value) || 0) / 100;
        dot({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }, '#eab308', 6);
      }
      draft.forEach(pt => dot(px(pt), '#facc15'));
      updateGeoStatus();
    }
    function updateGeoStatus() {
      const inc = geo.inclusion.length ? 'set' : 'none';
      const ln = (geo.line.p1 && geo.line.p2) ? 'set' : 'none';
      document.getElementById('geoStatus').textContent =
        `Inclusion: ${inc} | Exclusions: ${geo.exclusions.length} | Base line: ${ln} | Threshold: ${document.getElementById('threshold_point').value}`;
    }
    snapCanvas.addEventListener('click', (e) => {
      if (!mode) return;
      const r = snapCanvas.getBoundingClientRect();
      const pt = [ Math.round((e.clientX - r.left) / r.width * 1000) / 10,
                   Math.round((e.clientY - r.top) / r.height * 1000) / 10 ];
      draft.push(pt);
      if (mode === 'inclusion' && draft.length === 4) { geo.inclusion = draft.slice(); draft = []; mode = null; }
      else if (mode === 'exclusion' && draft.length === 4) { geo.exclusions.push(draft.slice()); draft = []; }  // stay armed for more zones
      else if (mode === 'line' && draft.length === 2) { geo.line = { p1: draft[0], p2: draft[1] }; draft = []; mode = null; }
      refreshTools(); redraw(); persistGeometry();
    });
    async function grabSnapshot() {
      document.getElementById('snapHint').textContent = 'Capturing snapshot...';
      try {
        const res = await fetch('/api/snapshot', { cache: 'no-store' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const blob = await res.blob();
        snapImg.onload = () => {
          snapCanvas.width = snapImg.naturalWidth;
          snapCanvas.height = snapImg.naturalHeight;
          document.getElementById('snapHint').style.display = 'none';
          snapImg.style.display = 'block';
          snapCanvas.style.display = 'block';
          refreshTools(); redraw();
        };
        snapImg.src = URL.createObjectURL(blob);
      } catch (err) {
        document.getElementById('snapHint').textContent = 'Snapshot failed: ' + err.message;
      }
    }
    function undoLast() {
      if (draft.length) draft.pop();
      else if (geo.line.p1) geo.line = { p1: null, p2: null };
      else if (geo.exclusions.length) geo.exclusions.pop();
      else if (geo.inclusion.length) geo.inclusion = [];
      refreshTools(); redraw(); persistGeometry();
    }
    function clearAllGeo() {
      geo = { inclusion: [], exclusions: [], line: { p1: null, p2: null } };
      draft = []; mode = null; refreshTools(); redraw(); persistGeometry();
    }
    async function persistGeometry() {
      try {
        await fetch('/api/geometry', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            inclusion_polygon: geo.inclusion,
            exclusion_polygons: geo.exclusions,
            base_line: { p1: geo.line.p1, p2: geo.line.p2 },
            threshold_point: parseFloat(document.getElementById('threshold_point').value) || 0
          })
        });
      } catch (e) {}
    }

    async function loadConfig() {
      const res = await fetch('/api/config');
      const cfg = await res.json();
      geo.inclusion = Array.isArray(cfg.inclusion_polygon) ? cfg.inclusion_polygon : [];
      geo.exclusions = Array.isArray(cfg.exclusion_polygons) ? cfg.exclusion_polygons : [];
      geo.line = (cfg.base_line && cfg.base_line.p1 && cfg.base_line.p2) ? { p1: cfg.base_line.p1, p2: cfg.base_line.p2 } : { p1: null, p2: null };
      document.getElementById('device_priority').value = cfg.device_priority || 'NPU,GPU,CPU';
      document.getElementById('rtsp_url').value = cfg.rtsp_url;
      document.getElementById('esp32_host').value = cfg.esp32_host || '';
      document.getElementById('threshold_point').value = cfg.threshold_point;
      document.getElementById('threshold_delay').value = cfg.threshold_delay ?? 0;
      document.getElementById('max_fps').value = cfg.max_fps ?? 10;
      document.getElementById('auto_start').checked = !!cfg.auto_start;
      document.getElementById('conf_threshold').value = cfg.conf_threshold ?? 0.70;
      document.getElementById('iou_match_threshold').value = cfg.iou_match_threshold ?? 0.30;
      document.getElementById('min_consecutive_frames').value = cfg.min_consecutive_frames ?? 3;
      document.getElementById('subsequent_conf_threshold').value = cfg.subsequent_conf_threshold ?? 0.50;
      document.getElementById('track_validity_ms').value = cfg.track_validity_ms ?? 500;
      const ps = cfg.position_smoothing;
      document.getElementById('position_smoothing').value =
        (typeof ps === 'number') ? ps : (ps === false ? 0 : 0.25);   // migrate old true/false
      document.getElementById('track_agility').value = cfg.track_agility ?? 0.5;
      document.getElementById('out1_trigger').value = cfg.outputs["1"].trigger;
      document.getElementById('out1_duration').value = cfg.outputs["1"].duration;

      let html = '';
      for (let i = 2; i <= 8; i++) {
        const o = cfg.outputs[i.toString()];
        html += `
          <div style="border-top: 1px solid #334155; margin-top: 10px; padding-top: 10px;">
            <h4>Output ${i}</h4>
            <label>Trigger Position</label>
            <input type="number" id="out_${i}_trigger" value="${o.trigger}" step="0.1">
            <label>Duration (s)</label>
            <input type="number" id="out_${i}_duration" value="${o.duration}" step="0.1" min="0.1">
            <label><input type="checkbox" id="out_${i}_depends" ${o.depends ? 'checked' : ''}> Depends on Main</label>
            <label>Max Activations (during Main)</label>
            <input type="number" id="out_${i}_max" value="${o.max_act}">
            <label><input type="checkbox" id="out_${i}_reverse" ${o.reverse ? 'checked' : ''}> Reverse Trigger (High-to-Low)</label>
          </div>
        `;
      }
      document.getElementById('outputsConfig').innerHTML = html;
      for (let i = 2; i <= 8; i++) {
        const rev = document.getElementById(`out_${i}_reverse`);
        const dep = document.getElementById(`out_${i}_depends`);
        const sync = () => {
          if (rev.checked) { dep.checked = true; dep.disabled = true; }
          else { dep.disabled = false; }
        };
        rev.addEventListener('change', sync);
        sync();  // reverse outputs always depend on Main
      }
      const tp = document.getElementById('threshold_point');
      tp.addEventListener('input', redraw);
      tp.addEventListener('change', persistGeometry);
      refreshTools(); redraw();
      grabSnapshot();  // show the stream + any saved geometry immediately on load
    }

    async function saveConfig() {
      const res = await fetch('/api/config');
      let cfg = await res.json();
      cfg.device_priority = document.getElementById('device_priority').value;
      cfg.rtsp_url = document.getElementById('rtsp_url').value;
      cfg.esp32_host = document.getElementById('esp32_host').value.trim();
      cfg.threshold_point = parseFloat(document.getElementById('threshold_point').value);
      cfg.threshold_delay = parseFloat(document.getElementById('threshold_delay').value) || 0;
      cfg.max_fps = parseFloat(document.getElementById('max_fps').value);
      cfg.auto_start = document.getElementById('auto_start').checked;
      cfg.conf_threshold = parseFloat(document.getElementById('conf_threshold').value);
      cfg.iou_match_threshold = parseFloat(document.getElementById('iou_match_threshold').value);
      cfg.min_consecutive_frames = parseInt(document.getElementById('min_consecutive_frames').value);
      cfg.subsequent_conf_threshold = parseFloat(document.getElementById('subsequent_conf_threshold').value) || 0;
      const vms = parseFloat(document.getElementById('track_validity_ms').value);
      cfg.track_validity_ms = isNaN(vms) ? 500 : vms;
      const ps = parseFloat(document.getElementById('position_smoothing').value);
      cfg.position_smoothing = isNaN(ps) ? 0.25 : Math.min(Math.max(ps, 0), 1);   // 0 is valid (raw)
      const ag = parseFloat(document.getElementById('track_agility').value);
      cfg.track_agility = isNaN(ag) ? 0.5 : Math.min(Math.max(ag, 0), 1);
      cfg.inclusion_polygon = geo.inclusion;
      cfg.exclusion_polygons = geo.exclusions;
      cfg.base_line = { p1: geo.line.p1, p2: geo.line.p2 };
      cfg.outputs["1"].trigger = parseFloat(document.getElementById('out1_trigger').value);
      cfg.outputs["1"].duration = parseFloat(document.getElementById('out1_duration').value);

      for (let i = 2; i <= 8; i++) {
        const reverse = document.getElementById(`out_${i}_reverse`).checked;
        cfg.outputs[i.toString()] = {
          trigger: parseFloat(document.getElementById(`out_${i}_trigger`).value),
          duration: parseFloat(document.getElementById(`out_${i}_duration`).value),
          depends: reverse ? true : document.getElementById(`out_${i}_depends`).checked,
          max_act: parseInt(document.getElementById(`out_${i}_max`).value),
          reverse: reverse
        };
      }

      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(cfg)
      });
      alert('Configuration saved successfully! OpenVINO reloading model...');
    }
    loadConfig();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def execute_page():
    return EXECUTE_HTML

@app.get("/execute", response_class=HTMLResponse)
async def execute_route():
    return EXECUTE_HTML

@app.get("/setup", response_class=HTMLResponse)
async def setup_route():
    return SETUP_HTML
