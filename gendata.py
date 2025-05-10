#!/usr/bin/env python3
import os
import json
import cv2
import random
import argparse
import re

# ─── USER CONFIG ───────────────────────────────────────────────────────────────
CLASSES = [
    "chemical-spill(s)",
    "liquid-spill(s)",
    "oil-spill(s)",
    "oil-stain(s)",
    "liquid-stain(s)",
]
SAMPLE_FRAC = 0.4   # 40% per class
# ────────────────────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    """Make a filesystem-safe ID out of a class name."""
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

def yolo_to_coco(xc, yc, wn, hn, img_w, img_h):
    """Convert YOLO x_center,y_center,w,h (normalized) → COCO x1,y1,x2,y2 (px)."""
    w_px = wn * img_w
    h_px = hn * img_h
    x1 = int(xc * img_w - w_px/2)
    y1 = int(yc * img_h - h_px/2)
    x2 = int(xc * img_w + w_px/2)
    y2 = int(yc * img_h + h_px/2)
    return [x1, y1, x2, y2]

def buil_system_prompt() -> str:
    return (
        "You are an adept Factory Indoors Leaks, Spills and Object Detection model. Strictly Avoid False Positives."
    )

def build_user_prompt(class_name: str) -> str:
    return (
        "<image>\n"
        f"Detect and return the bbox coordinates of the {class_name} "
        "in the image in COCO format if any."
    )

def main(args):
    random.seed(args.seed)
    img_files = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    ])
    n = len(img_files)
    sample_k = int(n * SAMPLE_FRAC)
    print(f"Found {n} images → sampling {sample_k} per class → total ≈ {sample_k*len(CLASSES)} entries")

    entries = []
    count = 0
    for cls in CLASSES:
        picks = random.sample(img_files, sample_k)
        system_prompt_text = buil_system_prompt()
        user_prompt_text = build_user_prompt(cls)
        slug = slugify(cls)

        for fn in picks:
            base = str(count).zfill(len(str(n))+1)
            img_path = os.path.join(args.images_dir, fn)
            lbl_path = os.path.join(args.labels_dir, fn[:-4] + ".txt")

            # load image to get dims
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] could not load {fn}, skipping")
                continue
            h, w = img.shape[:2]

            # parse YOLO labels (all class 0)
            dets = []

            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        _, xc, yc, wn, hn = map(float, parts)
                        box = yolo_to_coco(xc, yc, wn, hn, w, h)
                        dets.append({"bbox_2d": box, "label": cls})
            else:
                print(f"[WARN] no label for {fn}, skipping")
                continue
            count += 1
            # build JSON entry
            entry = {
                "id": f"{slug}_{base}",
                "messages": [
                    {
                        'role': 'system',
                        'content': [
                            {"type": "text", 
                            "text": system_prompt_text}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", 
                            "image": img_path, 
                            "min_pixels": 224 * 224, 
                            "max_pixels": 2048 * 2048},
                            
                            {"type": "text",  
                            "text": user_prompt_text}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text",
                            "text": f"{{\n\"{slug}\": {json.dumps(dets, indent=2)}\n}}"}
                        ],
                    },
                ]
            }
            entries.append(entry)

    # write out
    with open(args.output, 'w') as out:
        json.dump(entries, out, indent=2)
    print(f"Wrote {len(entries)} examples to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make few-shot LoRA JSON with 5 classes ×40% sampling")
    p.add_argument("--images_dir", default="/home/aeternum/SE/OVD/Spill Dataset/train/images",
                   help="Folder with your .jpg/.png images")
    p.add_argument("--labels_dir", default="/home/aeternum/SE/OVD/Spill Dataset/train/labels",
                   help="Folder with YOLO .txt labels (class 0 only)")
    p.add_argument("--output",     default="_training_data.json",
                   help="Path to write the JSON array")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for reproducibility")
    args = p.parse_args()
    main(args)
