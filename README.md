# YOLO-Halloween

An early-development Halloween trigger engine. Point a camera at your walkway, it watches people move through, and it trips relays as they hit your marks. Two pieces:

- **ESP32 firmware** — a tiny control app, an API, and UDP discovery. This is what actually drives the relays. Build instructions below. $20 max.
- **Docker backend** — Intel-optimized YOLO26m (OpenVINO) doing the detection, plus the tracking logic and web UI for setup and monitoring.

Optional: up to 4 WLED segments with creepy positional effects (use a separate controller for now).

This is a personal project. I got tired of re-teaching myself my old self-built manual rig every October — a dozen timers, pull-ups, beam sensors, and a rat's nest of wiring. This rig is self-explanatory: open `http://server:8777`, go to setup, draw a box and a line, and you're off to the races. The setup page explains every setting because you use this once a year and won't remember how it works.

**Status:** tested on a Meteor Lake 155H (NPU or iGPU), holds ~20 FPS.

## Features

- **SPDT dry contacts for everything.** Possibilities are endless.
- **SOTA edge inference.** YOLO26m @ 640x640 FP16 (OpenVINO) on NPU, GPU, or CPU. Each person gets a Kalman track.
- **Easy, reliable triggers.** Each detection is reduced to its box bottom-center, projected perpendicularly onto the base line you draw.
- **Zone filtering.** A 4-point inclusion polygon plus optional exclusion zones; a detection only counts if its box bottom-center passes both.
- **Robust trigger logic.** A trigger fires when a person crosses it *or* gets within 20 points in front of it (behind it, for reverse). It fires every time you have a remotely decent view, and you can tweak scenes without stopping the engine — as long as nobody comes within +20 points of the Main trigger.
- **8 distinct triggers (can drive 50 contacts).** Output 1 is **Main** (your scene) — it holds the show for its duration, and when it ends everything resets. Outputs 2-8 are the gadgets: each gets a trigger point, a duration, an optional "depends on Main", a max-activation cap, and an optional reverse mode (fires on the way back).
- **Threshold + delay.** Reverse outputs (which always depend on Main) stay disarmed until someone crosses a threshold point *and* a configurable delay passes.
- **Live tuning, no restart.** Confidence thresholds, Kalman position smoothing (0 = snappy, 1 = buttery) and track agility are all adjustable while it runs.
- **Web UI, zero manual JSON or YAML.** `/execute` shows a timeline canvas with stable-id dots, live telemetry, manual output buttons, and global off/reset (SSE at ~10 Hz). `/setup` holds all the config plus snapshot calibration — draw your polygons and line right on a grabbed frame.
- **Name your buttons.** Rename Outputs 2-8 on `/setup` and they show up on `/execute`. Main always reads "Main".
- **Docker + `/data` volume.** Config and model live in `./data`, so the container stays disposable.
- **Sane controller link.** One serialized TCP connection at a time (basic ESP32s wedge on concurrent clients), bounded timeouts, retries, an idle PING keep-alive, and a non-blocking command queue so a slow board can't stall inference.
- **WLED lighting (optional).** Creepy position-based effects, quadratic 3-point calibration, up to 4 pixel runs.

## Preview

**▶ [Preview video](preview/example.mp4)** — live tracking and relays firing.

Poor video quality, first time using Uniview prop at night, day/night/illumination not locked down hard enough and gain too high — still works great.

## Limitations

- **It's 1-D.** Everything collapses onto one base line (via that perpendicular projection) — it knows *how far along* someone is, not where they are in 2-D.
- **One camera, one line, one config.** Single RTSP stream, single base line, one global config. No multi-camera or multi-zone setups.
- **People only.** The detector only acknowledges the person class.
- **Fixed hardware shape.** 8 channels tied to a single ESP32 — not expandable without code changes.
- **10 tracked people max.** Extras get dropped (mainly affects WLED).
- **No auth.** Trusted-LAN only — don't expose port 8777 to the internet.
- **The UI is a mirror, not the truth.** It shows the *intended* output state (no GPIO read-back), and it plots the backend's Kalman output with no client-side smoothing — so bad WiFi means a stuttering dot.
- **Calibration is snapshot-based.** You draw geometry on a still frame, not live — move the camera and you must re-draw.
- **Some assembly required.** x86 Linux + Docker; the Intel NPU/GPU runtimes get pulled at image build time and the YOLO weights download on first start (`HALO_WEIGHTS` builds offline).
- **It can be fooled.** YOLO26m is excellent and worth the AGPL3 license, but Halloween scenes are chaotic — filter with caution and give the camera a clear view of the path.
- **Latency exists.** Tuned for low end-to-end delay, but camera + inference + network still add up. Best effort, with plenty of knobs to dial in the speed you want.

## Quick start

```
docker compose up -d
```

Then open **http://docker-host:8777** (the web UI) and:

1. **Calibrate** on `/setup` — set the RTSP URL, grab a snapshot, draw the inclusion zone and base line, and set the threshold (if using reverse). Name and tune Outputs 2-8 here too.
2. **Start the engine** on `/execute` (or tick *Auto-start engine on boot* on `/setup`). Nothing fires until it's running — the manual output buttons work regardless.
3. **Flash the ESP32** — paste `esp32/firmware.cpp` into the Arduino IDE (ESP32 core), set your WiFi SSID/password, and flash. It serves its own control page on port 80 and talks to the backend on TCP 5000 (UDP discovery on 5001). Pin map below.
4. **WLED (optional)** — on the WLED device enable *Receive Realtime State* / Network DDP, set transitions to **0 ms**, *Realtime Timeout* to 2-3 s, and *Force Max Brightness in Realtime*. On `/setup` tick **Enable WLED output**, enter the IP, and set up to 4 segments (pixel count + the line positions where that run's first and last pixels actually sit + one mid pixel). Device-side checklist below.

## Hardware

Too easy for a schematic, endless possibilities with NC + NO dry contacts. The backend runs OpenVINO in a Trixie container; the box is:

- **1x ESP32**, **1x darlington array**, and **1-2x 8-channel relay boards** — one darlington collector can drive several relays or opto isolators.
- That gives universal dry-contact triggers: quick scares, smoke machines (longer duration = more smoke), and active/idle scene switching on AC or DC (full lighting shifts, motors, strobes). I personally expose both NO and NC for 12 V and 120 V on **Main**.

Each ESP32 output goes to a darlington base (ULN2803A); the collectors drive active-low relay boards, so a HIGH GPIO activates the relay. Boards can be 5 V or 12 V as long as you supply that voltage — I use 12 V because the same enclosure feeds 6 V to Spirit animatronics that are finicky about 5 V inputs.

**Intel required (for now)** — any supported NPU, or an iGPU at least as new as Iris Xe.

**Use a commercial camera** — Dahua, Axis, Uniview, Hikvision, Hanwha, etc. — so you can tune exposure, gain and day/night, and get a **640-horizontal-pixel sub stream**.

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

## WLED setup

The app drives the strip with raw RGB over **DDP (UDP 4048)**, so WLED needs a few boxes ticked:

- **Kill the smoothing.** *Settings → LED Preferences* → transition time **0 ms**, palette transitions off. Otherwise WLED eases every change and the holes lag behind anyone moving quickly.
- **Accept realtime data.** *Settings → Sync Interfaces* → tick **Receive Realtime State**, leave **Network DDP** enabled (UDP 4048), set **Realtime Timeout** to 2-3 s, and tick **Force Max Brightness in Realtime** — skip that last one and WLED dims our values right back down.
- **Match the pixel counts.** WLED's segments should add up to the pixels you configured per run on `/setup`, and DDP pixel 0 is the first LED on the data pin. Run wired backwards? Tick that segment's **Reverse direction** in WLED — the app never worries about direction.

## Knobs

Everything's on `/setup`, saves to `./data/config.yml`, and most of it applies **live** (no restart). Four groups: what the detector believes, how the dot moves, what fires, and the optional WLED strip.

**Camera & system**
- **Device priority** — accelerator order (CPU is always the fallback). Changing it recompiles the model.
- **RTSP URL** and **ESP32 host** (blank = UDP auto-discovery).
- **Max inference FPS** — higher = fresher positions; stale frames are dropped automatically.
- **Auto-start engine on boot** — otherwise you press START ENGINE on `/execute`.

**Detection & tracking**
- **Person confidence** / **subsequent confidence** — min score to start a track, and the lower score allowed to *extend* one.
- **IoU match** — min box overlap to consider it the same person between frames.
- **Consecutive detections** — hits required before a track registers.
- **Track validity window (ms)** — how long a missed track coasts. Think in **frames**: 2–5 frames' worth is the sweet spot (default **200 ms**); too long and the dot extrapolates past a person who just stopped.

**Position filter (Kalman)** — the reported 0–100 is the *filtered* box bottom-center (drives the dot, the triggers, and the baseline).
- **Position smoothing (0–1)** — low = near-real-time (some jitter); high = steady but laggy.
- **Track agility (0–1)** — how fast the filter follows sudden speed changes; higher stays locked to fast moves but is jumpier.

**Triggers & outputs**
- **Threshold point** + **threshold delay** — where the reverse logic arms, and how long after the cross it waits.
- **Outputs 1–8** — enable/disable (2–8), trigger point, duration, depends-on-Main, max activations, reverse; plus **button names** for 2–8 (Main always reads "Main"). A disabled channel never fires and its button is hidden on `/execute`.

**WLED lighting (optional)** — its own card on `/setup`. Ships **enabled** but inert until you give it an IP, so nothing is ever sent to a strip you haven't named.
- **Enable WLED output** + **WLED IP** (manual; no discovery). **Test** sends one real DDP frame so you can confirm it lands.
- **While Main is OFF** — color + brightness (default deep orange-red `#CC3300` @ 50%).
- **While Main is ON** — **Scene** (`follower` = dark holes track people, `follower_narrow` for sparse runs, or `solid`; default follower), color (default deep red `#800000`; a scene may ignore it), brightness (default 50%), and **Refresh Rate** (Hz — default 15; keeps WLED in realtime and sets how fast the holes move). Pick the follower by pixel density: `follower` is a 5-pixel hole with 2-pixel rings (21-pixel shadow, ~60 px/m runs), `follower_narrow` a 1-pixel hole with 1-pixel rings (9-pixel shadow, ~15 px/m runs).
- **Segments** (up to 4) — each is a run of pixels along the line: **pixel count**, the line position of its **first** and **last** pixel, and one **mid** pixel (number + position, roughly midway). Those 3 points calibrate a quadratic pixel↔position map, so uneven spacing is handled. Pixel 0 is the run's first DDP pixel; physical direction is WLED's own segment "reverse" checkbox, not the app's.
- **Line span isn't 0-100 — and it isn't your pixel range.** A position is a perpendicular projection onto your base line, so wherever the inclusion zone runs past the line's ends it reads negative or past 100; `/setup` shows that outer boundary for whatever you've drawn. Don't feed it to the segments: the span is how far a *person* can read, while a run's first/last pixel positions are wherever those pixels physically sit — usually a narrower range inside it.

## Roadmap

(maybe this year, maybe next)

1. More WLED position effects.
2. Webhook outputs for Home Assistant or other web-based scares.
3. Additional or backup sensor inputs. (Questionable — Halloween went better than expected, I'm ditching the lasers for good.)
4. Manual push-button triggers without a position trigger (more check boxes).
5. Continue to improve tracking.
