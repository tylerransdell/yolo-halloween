# YOLO-haloween

A working-prototype Halloween trigger engine. Point a camera at your walkway, it watches
people move through, and it trips relays as they hit your marks. Two pieces:

1. **ESP32 firmware** — a tiny control app, an API, and UDP discovery. This is what actually drives the relays.
2. **Docker backend** — Intel-optimized YOLO26m (OpenVINO) doing the detection, plus the tracking logic and web UI that decide when to poke the ESP32.

This is a personal project. I got tired of re-teaching myself my old self-built manual rig
every October — a dozen timers, pull-ups, beam sensors, and a rat's nest of wiring.
Tested on a Meteor Lake 155H (NPU + iGPU), holds ~20 FPS.

## Features

- **Real person tracking.** YOLO26m @ 640x640 FP16 (OpenVINO) on NPU, GPU, or CPU. Each person gets a constant-velocity Kalman track with a stable ID, so fast movers and short dropouts stay the same person and two people crossing don't swap identities.
- **One number to trigger on.** Each detection is reduced to its box bottom-center, projected **perpendicularly** onto the base line you draw, and read off as 0-100. That single number is the person's "position", and every trigger just compares a point against it (someone is "past" a trigger when their perpendicular projection lands beyond it).
- **Zone filtering.** A 4-point inclusion polygon plus optional exclusion zones; a detection only counts if its box bottom-center passes both.
- **8 outputs.** Output 1 is **Main** (your scene) — it holds the show for its duration, and when it ends everything resets. Outputs 2-8 are the gadgets: each gets a trigger point, a duration, an optional "depends on Main", a max-activation cap, and an optional reverse mode (fires on the way back, high-to-low).
- **Threshold + delay.** Reverse outputs (which always depend on Main) stay disarmed until someone crosses a threshold point *and* a configurable delay passes.
- **Live tuning, no restart.** Confidence thresholds, Kalman position smoothing (0 = snappy, 1 = buttery) and track agility are all adjustable while it runs.
- **Web UI.** `/execute` shows a timeline canvas with stable-id dots, live telemetry, manual output buttons, and global off/reset. `/setup` holds all the config plus snapshot calibration — draw your polygons and line right on a grabbed frame. Pushed over SSE at ~10 Hz (300 ms poll fallback).
- **Name your buttons.** Rename Outputs 2-8 on `/setup` and they show up on `/execute`. Main always reads "Main".
- **Docker + `/data` volume.** Config and model live in `./data`, so the container stays disposable.
- **Sane controller link.** One serialized TCP connection at a time (basic ESP32s wedge on concurrent clients), bounded timeouts, retries, an idle PING keep-alive, and a non-blocking command queue so a slow board can't stall inference.

## Limitations

- **It's 1-D.** Everything collapses onto one base line (via that perpendicular projection) — it knows *how far along* someone is, not where they are in 2-D. Two people at the same point read as the same number.
- **One camera, one line, one config.** Single RTSP stream, single base line, one global config. No multi-camera or multi-zone setups.
- **People only.** The detector is tuned for the person class; cars, pets, and decorations don't count.
- **Fixed hardware shape.** 8 channels tied to a single ESP32 — not expandable without code changes.
- **10 tracked people max.** Extras get dropped.
- **The UI is a mirror, not the truth.** It shows the backend's Kalman output and the *intended* output state. There's no GPIO read-back, so a button can briefly show something the pin isn't doing.
- **No auth.** Trusted-LAN only — don't expose port 8777 to the internet.
- **Bad WiFi = a laggy dot.** The dot is exactly the backend's Kalman output with zero client-side smoothing, so a weak link makes it stutter, freeze, and jump. The polling fallback only kicks in if the browser can't do EventSource at all.
- **Calibration is snapshot-based.** You draw geometry on a still frame, not live — move the camera and you re-draw.
- **Some assembly required.** x86 Linux + Docker; the model IR isn't in the image (generate/mount it in `/data/models`) and the Intel NPU/GPU runtimes get pulled at build time.
- **It can be fooled.** YOLO26m is excellent and worth the AGPL3 license but haloween scenes can be chaotic, so filter with caution.
- **Latency exists.** Tuned for low end-to-end delay, but camera + inference + network still add up. This is a best effort application with plenty of knobs to dial in the speed you want.

## Quick start

```
docker compose up -d
```

Then open **http://docker-host:8777** (that's the web UI) and:

1. **Calibrate** on `/setup` — set the RTSP URL, grab a snapshot, draw the inclusion zone and base line, and set the threshold. Name and tune Outputs 2-8 here too.
2. **Start the engine** from `/execute` (or tick *Auto-start engine on boot* on `/setup`). Nothing fires automatically until it's running — the manual output buttons work regardless.
3. **Flash the ESP32** — paste `esp32/firmware.cpp` into the Arduino IDE (ESP32 core), set your WiFi SSID/password, and flash. It serves its own control page on port 80 and talks to the backend on TCP 5000 (UDP discovery on 5001). Pin map is in [ESP32 wiring](#esp32-wiring) below.
4. **Model** — the backend loads the OpenVINO IR from `/data/models/yolo26m`. Build it with `build_model.py`, or drop `yolo26m.pt` in `/data/models` and it converts on first load.

Hardware is cheap and easy if you already have an Intel inference box: 1x ESP32, 1x darlington array,
and 1-2x 8-channel relay boards (one darlington collector can drive several relays). That
gives you universal dry-contact triggers for quick scares, smoke machines (longer duration =
more smoke), and active/idle scene switching on AC or DC.

## ESP32 wiring

8 channels in a fixed GPIO order, **active HIGH** (a HIGH drives the relay on):

| Output | GPIO |
| --- | --- |
| 1 — Main | 32 |
| 2 | 33 |
| 3 | 25 |
| 4 | 26 |
| 5 | 27 |
| 6 | 18 |
| 7 | 19 |
| 8 | 23 |

Each output feeds a darlington input, and one darlington collector can drive several relay
coils — so you can fan a single channel out to multiple relays.

## Knobs

Everything's on `/setup`, saves to `./data/config.yml`, and most of it applies **live** (no
restart). Three groups: what the detector believes, how the dot moves, and what fires.

**Camera & system**
- **Device priority** — accelerator order (CPU is always the fallback). Changing it recompiles the model.
- **RTSP URL** and **ESP32 host** (blank = UDP auto-discovery).
- **Max inference FPS** — cap; higher = fresher positions, stale frames are dropped automatically.
- **Auto-start engine on boot** — otherwise you press START ENGINE on `/execute`.

**Detection & tracking**
- **Person confidence** — min score to count as a detection.
- **Subsequent confidence** — lower score allowed to *extend* an already-tracked person.
- **IoU match** — min box overlap to consider it the same person between frames.
- **Consecutive detections** — hits required before a track registers.
- **Track validity window (ms)** — how long a missed track coasts before it's dropped. Think in **frames**, not ms: **2–5 frames' worth** is the sweet spot (default **200 ms**), scaled by your FPS. Too long and the dot extrapolates past a person who just stopped (elastic overshoot).

**Position filter (Kalman)** — the reported 0–100 is the *filtered* box bottom-center (drives the dot, the triggers, and the baseline).
- **Position smoothing (0–1)** — low = near-real-time (some jitter); high = steady but laggy.
- **Track agility (0–1)** — how fast the filter follows sudden speed changes; higher stays locked to fast moves but is jumpier.

**Triggers & outputs**
- **Threshold point** + **threshold delay** — where the reverse logic arms, and how long after the cross it waits.
- **Outputs 1–8** — trigger point, duration, depends-on-Main, max activations, reverse; plus **button names** for 2–8 (Main always reads "Main").

## Roadmap

(maybe this year, maybe next)

1. WLED integration.
2. Additional webhook outputs.
3. Additional or backup sensor inputs.
