Halloween Automation Engine: Complete System Specification
1. System Architecture & Deployment
Containerization: Self-hosted Docker container running on Linux.

Storage: Persistent configuration (config.json), polygons, lines, and output settings are saved locally in a /data volume directory.

Web Interface: Single application instance running on one port (e.g., 8777) with cross-linked navigation between routes:

/setup: Configuration dashboard for cameras, calibration, geometry, and output parameters.

/execute (Default): Operational dashboard for real-time tracking, numerical coordinate feeds, global overrides, and interactive output control panels.

Camera Integration: Pulls static snapshots via OpenCV/FFmpeg for setup calibration. Runtime inference runs headlessly in the backend using YOLO26m (640x640, OpenVINO optimized for NPU/GPU per ~/model/plan.md).

2. Setup Page (/setup)
All system calibration and output parameters are configured here and saved to /data/config.json.

A. System & Header Options
Inference Target Selector: Toggle between NPU or GPU execution profiles.

Output Naming: Custom text labels for Outputs 1 through 8 (Note: Output 1 is permanently designated as Main, ignoring custom labels).

B. Spatial Geometry & Calibration (Drawn on Snapshot)
Snapshot Tool: Captures a static frame from the local RTSP stream for calibration.

Inclusion Polygon: User draws a 4-point irregular polygon defining the active detection boundary (dynamic UI animation while drawing).

Exclusion Polygons: Optional 4-point zones where detected targets are ignored.

Base Line: User draws a 2-point reference line across the detection area. Automatically extends to polygon boundaries, normalized to 0-100 with support for negative numbers or values >100 in extended zones.

Threshold Point: A specific normalized point along the line that must be crossed before the reverse outputs can arm. A configurable Threshold Delay (seconds) after the cross must also elapse before reverse logic becomes active.

C. Output Configuration Panel (Outputs 1–8)
Output 1 (Main):

Trigger position (normalized point along baseline).

Duration (Default: 180 seconds).

Outputs 2–8 (Non-Main):

Trigger position (normalized point along baseline).

Duration (Default: 1 second).

Depends on Main (Checkbox): Toggles dependency on Main's active state.

Max Activations: Enabled only if "Depends on Main" is checked (sets max allowed firings during Main's active window; default 1).

Reverse Trigger (reverse: true - Checkbox): Inverts trigger logic to fire on a high-to-low crossing (returning targets) rather than low-to-high. Reverse outputs always depend on Main (the dependency is forced on).

3. Execution Page (/execute)
The operational dashboard optimized for real-time Halloween night monitoring and manual testing.

A. Real-Time Tracking & Telemetry (SSE push, ~10 Hz)
Linear Model Visualization: Clean graphic representation (canvas) showing the baseline, output trigger markers, threshold marker, and a labeled, stable-id dot for each detected person.

Numerical Positions Feed: A list displaying the exact linear coordinates of all currently detected persons (up to 10 people), each tagged with its stable track id (e.g. "#1: 42.7"). Pushed over Server-Sent Events (/api/events) instead of polling, with a 300ms polling fallback if SSE is unavailable.

B. ESP32 Network & Status Bar
Connection Health: Displays connection status, current ESP32 IP address (discovered via UDP), last command sent, and the ESP32's OK response.

ESP32 Link Robustness: All controller traffic is serialized through a single short-lived TCP connection at a time (a basic ESP32 wedges on concurrent clients). Each exchange is bounded by socket timeouts, always closes cleanly, retries once, and an idle keep-alive PING loop keeps the reported connection status honest.

Global Controls:

Off: Sets all outputs to zero (LOW) and halts detection loops.

Reset: Resets all active timers, clears output states, and resets threshold flags.

C. Output Interactive Control Panel (Outputs 1–8)
Displays real-time status indicators and manual trigger buttons for all 8 channels.

Dynamic UI Gating (Outputs 2–8): Buttons automatically grey out and disable if:

Main is not currently active.

The output has already reached its configured max activations limit during the current Main window.

The output is currently active (locked out for the duration of its timer).

4. Automation Finite State Machine (FSM) & Timing Rules
A. Global Reset & Threshold Logic
Arming: The threshold is crossed the first time at least one person is detected at or past the threshold point.

Threshold Delay: A configurable delay (seconds). Reverse logic stays inactive until this delay has elapsed after the threshold cross.

Reverse Outputs: Always depend on Main -- they can only arm and fire while Main is active.

Global Reset: Triggered only when Main's duration expires, or when the duration of any output that does not depend on Main expires. A full reset stops all timers, drops all outputs off, and clears the threshold/reverse state.

B. Output 1 (Main) Execution Rules
Trigger: Activated when a target is detected at or past the trigger point (low-to-high cross) AND still within the nearness band of that point (see Activation Gate below).

Constraints: Can only be activated once per window while running. Timers do not reset if additional people cross while Main is active.

Expiration: When Main's duration expires -- or when the duration of any output that does not depend on Main expires -- a full system reset is triggered (all timers stop, all outputs turn off, threshold/reverse state clears).

C. Outputs 2–8 Execution Rules
Trigger Logic: Standard low-to-high cross, unless Reverse Trigger is checked (which fires when the highest position person moves from above the trigger to below the trigger, remaining within the nearness band of the point). Reverse outputs always depend on Main.

Activation Gate: An output point is only satisfied when a detected position is at/past (forward) or at/below (reverse) the point AND within 20 normalized points of it. This prevents a trigger firing from the far side of the point (bounds Main especially).

Strict Duration Lockout: No output can reactivate within its own duration window. Once fired, it is locked out until its active duration fully elapses.

Main Dependency: If dependent on Main, it obeys its max activation ceiling and cannot fire outside Main's active window.

extra notes... important inference optimized openvino. end to end removed on yolo26m f16 640x640.  also snapshot should eb 640x480.  preferably this is stream received but if not it needs scaled to this. 

Every detected bounding box is evaluated using its bottom center point:

Inclusion Polygon: If defined, the bottom center point must lie inside the main inclusion polygon. If outside, it is discarded.

Exclusion Polygons: If any exclusion polygons are defined, any bottom center point falling inside them is filtered out.

Baseline Projection: Only valid points passing both filters are projected onto the baseline (0-100) and fed into the FSM.

Detection Confirmation (motion-tracked): Each track runs a constant-velocity Kalman filter. Every frame the track PREDICTS its box forward to the current timestamp, and association matches each detection to the track whose predicted box overlaps it best (greedy by descending IoU) -- so fast movers and brief dropouts stay on the correct track and two people crossing do not swap identities. A Confidence Threshold detection seeds a track; once seeded, ANY detection at or above the Subsequent Confidence Threshold may extend it. A track registers after the configured number of consecutive detections, keeps coasting on its prediction for the Track Validity Window (milliseconds without a detection) before being dropped, and each match updates (and re-predicts) the box for the next frame. The reported position IS the Kalman-filtered box bottom-center (the filter doubles as the smoother; there is no separate filter stage). It is tuned live by two knobs -- Position Smoothing (0 = near-real-time / lowest latency, 1 = heavy smoothing; maps to the measurement noise R) and Track Agility (how fast the filter follows sudden speed changes; maps to the velocity process noise Q). Each person carries a stable id shown in the UI.


extra notes: only scale until largest direction is 640 pixels... if input stream is 640x360... can just run inference on that because it fits in 640x640 detector.
