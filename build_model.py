#!/usr/bin/env python3
"""Build YOLO26m -> 640x640 FP16 OpenVINO IR (e2e / Top-K stripped).

ONE IR is produced and shared by every accelerator (NPU/GPU/CPU): the raw-head FP16 model
is the fastest on all of them, so don't fork a per-device build. (A GPU/e2e variant with the
Top-K/NMS head measured ~1.4x slower on the Arc GPU than this raw-head IR.)

Produces ``yolo26m.xml`` + ``yolo26m.bin`` inside ``--out`` (default
``/data/models/yolo26m``), matching ``main.py``'s ``MODEL_DIR``.

Run inside the container, or with the build venv:

    /home/tyler/model/yolo_env/bin/python build_model.py \
        --weights /home/tyler/model/yolo26m.pt \
        --out /home/tyler/halo/data/models/yolo26m
"""

import argparse
import glob
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description="YOLO26m -> 640 FP16 OpenVINO IR")
    ap.add_argument("--weights", default="yolo26m.pt", help="path to yolo26m.pt")
    ap.add_argument("--out", default=os.environ.get("MODEL_DIR", "/data/models/yolo26m"),
                    help="output directory for the .xml/.bin")
    ap.add_argument("--name", default="yolo26m", help="output model stem")
    ap.add_argument("--imgsz", type=int, default=640, help="square inference size")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"[build] weights not found: {args.weights}")
        sys.exit(1)

    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    print(f"[build] exporting {args.weights} -> OpenVINO IR @ {args.imgsz} FP16 (end2end=False)")

    model = YOLO(args.weights)
    export_path = model.export(
        format="openvino",
        imgsz=args.imgsz,
        dynamic=False,
        quantize=16,     # FP16 weights (compress_to_fp16)
        batch=1,
        nms=False,       # keep raw logits (no embedded NMS)
        end2end=False,   # strip the YOLO26 Top-K/Gather/NMS head -> NPU friendly
    )

    src_dir = os.path.dirname(str(export_path))
    for ext in (".xml", ".bin"):
        src = os.path.join(src_dir, f"{args.name}{ext}")
        if not os.path.exists(src):
            matches = glob.glob(os.path.join(src_dir, f"*{ext}"))
            src = matches[0] if matches else src
        dst = os.path.join(args.out, f"{args.name}{ext}")
        shutil.copy2(src, dst)
        print(f"[build] wrote {dst}")

    print("[build] done")


if __name__ == "__main__":
    main()
