# YOLO-halloween

A working-prototype Halloween trigger engine. Point a camera at your walkway, it watches
people move through, and it trips relays as they hit your marks. Two pieces:

1. **ESP32 firmware** — a tiny control app, an API, and UDP discovery. This is what actually drives the relays.
2. **Docker backend** — Intel-optimized YOLO26m (OpenVINO) doing the detection, plus the tracking logic and web UI that decide when to poke the ESP32.

Optional: up to 4 WLED segments with positional effects.

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
- **WLED lighting (optional).** Drive an addressable LED run as a *live* position line over **DDP** (UDP 4048). While Main is OFF the run holds one color/brightness; while Main is ON it renders a scene — **solid**, or the **follower** ("dimming") effect, where the pixels over each person go dark and step back up (zero, then 0.15×, 0.20×, 0.30×, 0.60×, then full) either side, overlaps keeping the darker value. The ramp is deliberately low and non-linear because perceived brightness isn't linear. Up to 4 pixel runs, each calibrated with a 3-point **quadratic** (evenly-spaced assumptions won't cut it — the app's 0-100 axis doesn't scale linearly with distance). Streamed off the inference path on its own timer.

## Limitations

- **It's 1-D.** Everything collapses onto one base line (via that perpendicular projection) — it knows *how far along* someone is, not where they are in 2-D. Two people at the same point read as the same number.
- **One camera, one line, one config.** Single RTSP stream, single base line, one global config. No multi-camera or multi-zone setups.
- **People only.** The detector is tuned for the person class; cars, pets, and decorations don't count.
- **Fixed hardware shape.** 8 channels tied to a single ESP32 — not expandable without code changes.
- **The WLED run needs calibrating by hand.** One strip per config, and it only lines up after you punch in each run's pixel count and a few pixel positions — the app's position number doesn't scale linearly to physical distance, so a straight 0-100 guess will be off at the ends.
- **10 tracked people max.** Extras get dropped.
- **The UI is a mirror, not the truth.** It shows the backend's Kalman output and the *intended* output state. There's no GPIO read-back, so a button can briefly show something the pin isn't doing.
- **No auth.** Trusted-LAN only — don't expose port 8777 to the internet.
- **Bad WiFi = a laggy dot.** The dot is exactly the backend's Kalman output with zero client-side smoothing, so a weak link makes it stutter, freeze, and jump. The polling fallback only kicks in if the browser can't do EventSource at all.
- **Calibration is snapshot-based.** You draw geometry on a still frame, not live — move the camera and you re-draw.
- **Some assembly required.** x86 Linux + Docker; the Intel NPU/GPU runtimes get pulled at image build time, and the YOLO weights are downloaded once on first start (needs internet that once — set `HALO_WEIGHTS` to build offline).
- **It can be fooled.** YOLO26m is excellent and worth the AGPL3 license but halloween scenes can be chaotic, so filter with caution.
- **Latency exists.** Tuned for low end-to-end delay, but camera + inference + network still add up. This is a best effort application with plenty of knobs to dial in the speed you want.

## Quick start

```
docker compose up -d
```

Then open **http://docker-host:8777** (that's the web UI) and:

1. **Calibrate** on `/setup` — set the RTSP URL, grab a snapshot, draw the inclusion zone and base line, and set the threshold. Name and tune Outputs 2-8 here too.
2. **Start the engine** from `/execute` (or tick *Auto-start engine on boot* on `/setup`). Nothing fires automatically until it's running — the manual output buttons work regardless.
3. **Flash the ESP32** — paste `esp32/firmware.cpp` into the Arduino IDE (ESP32 core), set your WiFi SSID/password, and flash. It serves its own control page on port 80 and talks to the backend on TCP 5000 (UDP discovery on 5001). Pin map is in [ESP32 wiring](#esp32-wiring) below.
4. **Model** — nothing to do. The backend loads the IR from `/data/models/yolo26m`; if it's missing it downloads `yolo26m.pt` once, builds the 640×640 FP16 IR into `/data/models/yolo26m`, then discards the weights (only the `.xml`/`.bin` persist). Set `HALO_WEIGHTS` to build from a local `.pt` instead (offline).
5. **WLED (optional)** — on the WLED device enable *Receive Realtime State* / Network DDP, set transitions to **0 ms**, *Realtime Timeout* to 2-3 s, and *Force Max Brightness in Realtime*. On `/setup` tick **Enable WLED output**, enter the IP, and set up to 4 segments (pixel count + the line position of the first/last pixel + one mid pixel) — hit **Set First/Last from line span** first so the ends line up. Device-side checklist is in [WLED setup](#wled-setup) below.

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

## WLED setup

The app drives the strip with raw RGB over **DDP (UDP 4048)**, so WLED needs a few boxes ticked:

- **Kill the smoothing.** *Settings → LED Preferences* → set **transition time (fade) to 0 ms**, and
  uncheck palette transitions. Otherwise WLED eases every change and the holes lag behind anyone
  moving quickly.
- **Accept realtime data.** *Settings → Sync Interfaces* → tick **Receive Realtime State**, leave
  **Network DDP** enabled (UDP 4048), set **Realtime Timeout to 2-3 s**, and tick **Force Max
  Brightness in Realtime** — skip that last one and WLED dims the values we send right back down.
- **Match the pixel counts.** The segments set up in WLED should add up to the same pixels you
  configured per run on `/setup`, and DDP pixel 0 is the first LED on the data pin. A run wired
  backwards? Tick that segment's **Reverse direction** in WLED — the app never worries about direction.

## Knobs

Everything's on `/setup`, saves to `./data/config.yml`, and most of it applies **live** (no
restart). Four groups: what the detector believes, how the dot moves, what fires, and the
optional WLED strip.

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
- **Outputs 1–8** — enable/disable (2–8), trigger point, duration, depends-on-Main, max activations, reverse; plus **button names** for 2–8 (Main always reads "Main"). A disabled channel never fires and its button is hidden on `/execute`.

**WLED lighting (optional)** — its own card on `/setup`. Ships **enabled** but inert until you give it an IP, so nothing is ever sent to a strip you haven't named.
- **Enable WLED output** + **WLED IP** (manual; no discovery). **Test** sends one real DDP frame so you can confirm it lands.
- **While Main is OFF** — color + brightness (default deep orange-red `#CC3300` @ 50%).
- **While Main is ON** — **Scene** (`follower` = dark holes track people, or `solid`; default follower), color (default deep red `#800000`; a scene may ignore it), brightness (default 50%), and **Refresh Rate** (Hz — default 15; keeps WLED in realtime and sets how fast the holes move).
- **Segments** (up to 4) — each is a run of pixels along the line: **pixel count**, the line position of its **first** and **last** pixel, and one **mid** pixel (number + position, roughly mid-span). Those 3 points calibrate a quadratic pixel↔position map, so uneven spacing is handled. Pixel 0 is the run's first DDP pixel; physical direction is WLED's own segment "reverse" checkbox, not the app's.
- **Line span isn't 0-100.** A position is a perpendicular projection onto your base line, so wherever the inclusion zone runs past the line's ends it reads negative or past 100. `/setup` shows the real span for whatever you've drawn, and **Set First/Last from line span** drops those numbers into the segments — use them, or the holes drift at the ends.

## Roadmap

(maybe this year, maybe next)

1. More WLED position effects.
2. Additional webhook outputs.
3. Additional or backup sensor inputs.
