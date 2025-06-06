#!/usr/bin/env python3
import os
import json
import cv2
import random
import argparse
import re

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_dataset_config(config_params, dataset_name=None):
    """Get dataset configuration from config file"""
    dataset_config = config_params.get('dataset_config', {})
    if dataset_name is None:
        dataset_name = dataset_config.get('dataset_name', 'spill_dataset')
    
    base_path = dataset_config.get('base_path', '/home/aeternum/Research2/SE/OVD')
    datasets = dataset_config.get('datasets', {})
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {list(datasets.keys())}")
    
    dataset_info = datasets[dataset_name]
    
    # Build full paths
    full_paths = {}
    for key, relative_path in dataset_info.items():
        if key.endswith('_images') or key.endswith('_labels'):
            full_paths[key] = os.path.join(base_path, relative_path)
        else:
            full_paths[key] = relative_path
    
    return full_paths

# ─── USER CONFIG ───────────────────────────────────────────────────────────────
DEFAULT_CLASSES = [
    "chemical-spill(s)",
    "chemical-leak(s)",
    "liquid-spill(s)",
    "oil-spill(s)",
    "oil-stain(s)",
    "liquid-stain(s)",
]

# Pluralistic classes for se_spill_dataset (only for spill-leak-stain class)
SE_SPILL_PLURALISTIC_CLASSES = [
    "outdoor-water",           # Class 0 - single prompt (no pluralistic)
    "chemical-spill(s)",       # Class 1 - pluralistic prompts for spill-leak-stain
    "chemical-leak(s)",        # Class 1
    "liquid-spill(s)",         # Class 1  
    "liquid-stain(s)",         # Class 1
    "oil-spill(s)",           # Class 1
    "oil-stain(s)",           # Class 1
]

# Mixed dataset configuration for combining se_spill_dataset + spill_dataset
MIXED_DATASET_CONFIG = {
    "primary_dataset": "se_spill_dataset",     # Your proprietary data
    "secondary_dataset": "spill_dataset",     # Public data for generalization
    "mixing_ratios": {
        # Ratio of secondary to primary data per class
        "outdoor-water": 0.3,
        "spill-leak-stain": 0.5,
    },
    "class_alignment": {
        # How to align public dataset classes with your dataset classes
        "spill_dataset": {
            "source_class_id": 0,      # spill_dataset class 0 (Spill)
            "target_class_id": 1,      # maps to se_spill_dataset class 1 (spill-leak-stain)
            "target_class_name": "spill-leak-stain"
        }
    }
}

SAMPLE_FRAC = 0.4   # 40% per class
# ────────────────────────────────────────────────────────────────────────────────

def get_pluralistic_classes(dataset_name, use_config_classes, dataset_config):
    """Get the appropriate classes for pluralistic training based on dataset"""
    if not use_config_classes:
        if dataset_name == "se_spill_dataset":
            return SE_SPILL_PLURALISTIC_CLASSES
        else:
            return DEFAULT_CLASSES
    else:
        # Use config classes for other datasets, but apply pluralistic logic for se_spill_dataset
        if dataset_name == "se_spill_dataset":
            return SE_SPILL_PLURALISTIC_CLASSES
        else:
            class_mapping = dataset_config.get('class_mapping', {})
            return list(class_mapping.values()) if class_mapping else DEFAULT_CLASSES

def get_class_id_for_prompt(prompt_name, dataset_name, dataset_config):
    """Map prompt name to correct class ID based on dataset"""
    if dataset_name == "se_spill_dataset":
        if prompt_name == "outdoor-water":
            return 0
        else:
            # All other prompts map to spill-leak-stain (class 1)
            return 1
    else:
        # For spill_dataset, all prompts map to class 0
        return 0

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

def get_mixed_dataset_classes(config_params, use_mixed_dataset=False):
    """Get classes for mixed dataset training"""
    if not use_mixed_dataset:
        return None
    
    # For mixed dataset, use se_spill_dataset classes as base
    primary_config = get_dataset_config(config_params, MIXED_DATASET_CONFIG["primary_dataset"])
    return SE_SPILL_PLURALISTIC_CLASSES

def load_mixed_dataset_images(config_params, class_name, class_id, sample_k, public_data_ratio=0.5):
    """Load and mix images from both datasets based on scaling ratios"""
    primary_dataset = MIXED_DATASET_CONFIG["primary_dataset"]
    secondary_dataset = MIXED_DATASET_CONFIG["secondary_dataset"]
    
    # Get dataset configs
    primary_config = get_dataset_config(config_params, primary_dataset)
    secondary_config = get_dataset_config(config_params, secondary_dataset)
    
    # Load ALL primary dataset images (no sampling)
    primary_images_dir = primary_config['train_images']
    primary_labels_dir = primary_config['train_labels']
    primary_img_files = [f for f in os.listdir(primary_images_dir) 
                        if f.lower().endswith(('.jpg','.jpeg','.png'))]
    
    # Use ALL primary data
    primary_samples = len(primary_img_files)
    
    # Scale public data based on primary data size
    if class_id == 0:
        # outdoor-water class - but public dataset doesn't have this, so no public data
        secondary_samples = 0
        mixing_info = "0.0% (outdoor-water not in public dataset)"
    elif class_id == 1:
        # All spill-related classes: add public data scaled by the ratio
        secondary_samples = int(primary_samples * public_data_ratio)
        mixing_info = f"{public_data_ratio:.1%} scaling ({secondary_samples} public samples)"
    else:
        # Other classes: add public data scaled by the ratio
        secondary_samples = int(primary_samples * public_data_ratio)
        mixing_info = f"{public_data_ratio:.1%} scaling ({secondary_samples} public samples)"
    
    print(f"  Loading mixed data for '{class_name}' (class {class_id}) - {mixing_info}")
    print(f"    Primary ({primary_dataset}): {primary_samples} samples (ALL proprietary data)")
    print(f"    Secondary ({secondary_dataset}): {secondary_samples} samples")
    
    mixed_data = []
    
    # Add ALL primary dataset samples
    for fn in primary_img_files:
        mixed_data.append({
            'filename': fn,
            'dataset': primary_dataset,
            'images_dir': primary_images_dir,
            'labels_dir': primary_labels_dir,
            'expected_class_id': class_id,
            'source_class_id': class_id  # Same as expected for primary dataset
        })
    
    # Add scaled secondary dataset samples if needed and class_id == 1 (spill classes only)
    if secondary_samples > 0 and class_id == 1:
        secondary_images_dir = secondary_config['train_images']
        secondary_labels_dir = secondary_config['train_labels']
        secondary_img_files = [f for f in os.listdir(secondary_images_dir) 
                              if f.lower().endswith(('.jpg','.jpeg','.png'))]
        
        # Get class alignment info
        alignment = MIXED_DATASET_CONFIG["class_alignment"][secondary_dataset]
        source_class_id = alignment["source_class_id"]
        
        # Sample the required number of secondary images (with replacement if needed)
        if secondary_samples <= len(secondary_img_files):
            secondary_picks = random.sample(secondary_img_files, secondary_samples)
        else:
            # If we need more samples than available, sample with replacement
            secondary_picks = random.choices(secondary_img_files, k=secondary_samples)
        
        for fn in secondary_picks:
            mixed_data.append({
                'filename': fn,
                'dataset': secondary_dataset,
                'images_dir': secondary_images_dir,
                'labels_dir': secondary_labels_dir,
                'expected_class_id': class_id,
                'source_class_id': source_class_id  # Original class ID in secondary dataset
            })
    
    return mixed_data

def load_validation_data_from_folders(config_params, dataset_name, classes, use_config_classes, sample_k=None):
    """Load validation data from dedicated validation folders if available"""
    try:
        dataset_config = get_dataset_config(config_params, dataset_name)
        
        # Check if validation folders exist
        if 'valid_images' not in dataset_config:
            return None, "No validation folders found in dataset config"
        
        valid_images_dir = dataset_config['valid_images']
        # Construct validation labels path (similar to how train_labels relates to train_images)
        valid_labels_dir = dataset_config.get('valid_labels', valid_images_dir.replace('/images', '/labels'))
        
        if not os.path.exists(valid_images_dir):
            return None, f"Validation images folder not found: {valid_images_dir}"
        
        print(f"Using dedicated validation folders:")
        print(f"  Images: {valid_images_dir}")
        print(f"  Labels: {valid_labels_dir}")
        
        # Load validation images
        img_files = sorted([
            f for f in os.listdir(valid_images_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        
        if not img_files:
            return None, f"No images found in validation folder: {valid_images_dir}"
        
        print(f"Found {len(img_files)} validation images")
        
        # If sample_k is specified, sample from validation set
        if sample_k and sample_k < len(img_files):
            img_files = random.sample(img_files, sample_k)
            print(f"Sampled {sample_k} validation images")
        
        val_entries = []
        count = 0
        
        for cls in classes:
            slug = slugify(cls)
            
            for fn in img_files:
                base = str(count).zfill(len(str(len(img_files)))+1)
                img_path = os.path.join(valid_images_dir, fn)
                lbl_path = os.path.join(valid_labels_dir, fn[:-4] + ".txt")

                # load image to get dims
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] could not load {fn} from validation set, skipping")
                    continue
                h, w = img.shape[:2]

                # parse YOLO labels
                dets = []

                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            class_id, xc, yc, wn, hn = map(float, parts)
                            class_id = int(class_id)
                            
                            # Get the expected class ID for this prompt
                            expected_class_id = get_class_id_for_prompt(cls, dataset_name, dataset_config)
                            
                            # Only include detections that match the expected class for this prompt
                            if class_id == expected_class_id:
                                box = yolo_to_coco(xc, yc, wn, hn, w, h)
                                dets.append({"bbox_2d": box, "label": cls})
                else:
                    print(f"[WARN] no label for validation image {fn}, skipping")
                    continue
                
                count += 1
                # build JSON entry
                entry = {
                    "id": f"{slug}_val_{base}",
                    "conversations": [
                        {
                            "from": "human", 
                            "value": f"<image>\nDetect and return the bbox coordinates of the {cls} in the image in COCO format if any."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(dets, indent=2)
                        }
                    ],
                    "image": img_path
                }
                val_entries.append(entry)
        
        return val_entries, f"Loaded {len(val_entries)} validation examples from dedicated folders"
        
    except Exception as e:
        return None, f"Error loading validation from folders: {str(e)}"

def load_mixed_validation_data_from_folders(config_params, classes):
    """Load validation data for mixed dataset from dedicated validation folders"""
    try:
        primary_dataset = MIXED_DATASET_CONFIG["primary_dataset"]
        secondary_dataset = MIXED_DATASET_CONFIG["secondary_dataset"]
        
        # Get dataset configs
        primary_config = get_dataset_config(config_params, primary_dataset)
        secondary_config = get_dataset_config(config_params, secondary_dataset)
        
        # Check if both datasets have validation folders
        if 'valid_images' not in primary_config:
            return None, f"No validation folder found for primary dataset: {primary_dataset}"
        if 'valid_images' not in secondary_config:
            return None, f"No validation folder found for secondary dataset: {secondary_dataset}"
        
        # Load primary validation data
        primary_valid_images_dir = primary_config['valid_images']
        primary_valid_labels_dir = primary_config.get('valid_labels', primary_valid_images_dir.replace('/images', '/labels'))
        
        primary_img_files = [f for f in os.listdir(primary_valid_images_dir) 
                            if f.lower().endswith(('.jpg','.jpeg','.png'))]
        
        # Load secondary validation data  
        secondary_valid_images_dir = secondary_config['valid_images']
        secondary_valid_labels_dir = secondary_config.get('valid_labels', secondary_valid_images_dir.replace('/images', '/labels'))
        
        secondary_img_files = [f for f in os.listdir(secondary_valid_images_dir)
                              if f.lower().endswith(('.jpg','.jpeg','.png'))]
        
        print(f"Validation data from dedicated folders:")
        print(f"  {primary_dataset}: {len(primary_img_files)} images")
        print(f"  {secondary_dataset}: {len(secondary_img_files)} images")
        
        val_entries = []
        count = 0
        
        for cls in classes:
            slug = slugify(cls)
            class_id = get_class_id_for_prompt(cls, "se_spill_dataset", None)
            
            # Process primary dataset validation images
            for fn in primary_img_files:
                base = str(count).zfill(len(str(len(primary_img_files) + len(secondary_img_files)))+1)
                img_path = os.path.join(primary_valid_images_dir, fn)
                lbl_path = os.path.join(primary_valid_labels_dir, fn[:-4] + ".txt")

                # load image to get dims
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] could not load {fn} from {primary_dataset} validation, skipping")
                    continue
                h, w = img.shape[:2]

                # parse YOLO labels
                dets = []

                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            label_class_id, xc, yc, wn, hn = map(float, parts)
                            label_class_id = int(label_class_id)
                            
                            # Check if this detection matches what we expect for this prompt
                            if label_class_id == class_id:
                                box = yolo_to_coco(xc, yc, wn, hn, w, h)
                                dets.append({"bbox_2d": box, "label": cls})
                else:
                    print(f"[WARN] no label for {fn} from {primary_dataset} validation, skipping")
                    continue
                
                count += 1
                # build JSON entry with dataset info  
                entry = {
                    "id": f"{slug}_val_{base}_{primary_dataset}",
                    "source_dataset": primary_dataset,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nDetect and return the bbox coordinates of the {cls} in the image in COCO format if any."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(dets, indent=2)
                        }
                    ],
                    "image": img_path
                }
                val_entries.append(entry)
            
            # Process secondary dataset validation images (only for spill classes)
            if class_id == 1:  # Only for spill-related classes
                alignment = MIXED_DATASET_CONFIG["class_alignment"][secondary_dataset]
                source_class_id = alignment["source_class_id"]
                
                for fn in secondary_img_files:
                    base = str(count).zfill(len(str(len(primary_img_files) + len(secondary_img_files)))+1)
                    img_path = os.path.join(secondary_valid_images_dir, fn)
                    lbl_path = os.path.join(secondary_valid_labels_dir, fn[:-4] + ".txt")

                    # load image to get dims
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[WARN] could not load {fn} from {secondary_dataset} validation, skipping")
                        continue
                    h, w = img.shape[:2]

                    # parse YOLO labels
                    dets = []

                    if os.path.exists(lbl_path):
                        with open(lbl_path) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                label_class_id, xc, yc, wn, hn = map(float, parts)
                                label_class_id = int(label_class_id)
                                
                                # Check if this detection matches what we expect for this prompt
                                if label_class_id == source_class_id:
                                    box = yolo_to_coco(xc, yc, wn, hn, w, h)
                                    dets.append({"bbox_2d": box, "label": cls})
                    else:
                        print(f"[WARN] no label for {fn} from {secondary_dataset} validation, skipping")
                        continue
                    
                    count += 1
                    # build JSON entry with dataset info  
                    entry = {
                        "id": f"{slug}_val_{base}_{secondary_dataset}",
                        "source_dataset": secondary_dataset,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\nDetect and return the bbox coordinates of the {cls} in the image in COCO format if any."
                            },
                            {
                                "from": "gpt",
                                "value": json.dumps(dets, indent=2)
                            }
                        ],
                        "image": img_path
                    }
                    val_entries.append(entry)
        
        return val_entries, f"Loaded {len(val_entries)} mixed validation examples from dedicated folders"
        
    except Exception as e:
        return None, f"Error loading mixed validation from folders: {str(e)}"

def main(args):
    random.seed(args.seed)
    
    # Load configuration
    config_params = load_config(args.config_path)
    
    # Determine training mode
    if args.use_mixed_dataset:
        # Validate public_data_ratio
        if args.public_data_ratio < 0.0:
            raise ValueError(f"public_data_ratio must be non-negative, got {args.public_data_ratio}")
        
        # Mixed dataset mode - combine se_spill_dataset + spill_dataset
        print("=== MIXED DATASET TRAINING MODE ===")
        dataset_name = "mixed_dataset"
        CLASSES = get_mixed_dataset_classes(config_params, True)
        
        print(f"Primary dataset: {MIXED_DATASET_CONFIG['primary_dataset']}")
        print(f"Secondary dataset: {MIXED_DATASET_CONFIG['secondary_dataset']}")
        print(f"Public data ratio: {args.public_data_ratio:.1%}")
        print(f"Using pluralistic classes: {CLASSES}")
        
        # Try to load validation from dedicated folders first
        if args.validation_split > 0:
            print(f"Validation approach: {'Dedicated folders' if not args.force_random_split else 'Random split from training data'}")
            
            validation_entries = None
            if not args.force_random_split:
                validation_entries, val_msg = load_mixed_validation_data_from_folders(config_params, CLASSES)
                print(f"Validation status: {val_msg}")
        
        # For mixed dataset, we use ALL proprietary data + scaled public data
        primary_config = get_dataset_config(config_params, MIXED_DATASET_CONFIG['primary_dataset'])
        primary_img_files = os.listdir(primary_config['train_images'])
        n = len([f for f in primary_img_files if f.lower().endswith(('.jpg','.jpeg','.png'))])
        sample_k = None  # Not used for mixed dataset
        print(f"Primary dataset images: {n} → using ALL proprietary data + {args.public_data_ratio:.1%} scaled public data")
        
        # For mixed dataset, we'll handle image/label dirs differently
        images_dir = None
        labels_dir = None
        
    else:
        # Single dataset mode (original logic)
        dataset_config = get_dataset_config(config_params, args.dataset_name)
        dataset_name = args.dataset_name or config_params['dataset_config']['dataset_name']
        
        # Get pluralistic classes based on dataset
        CLASSES = get_pluralistic_classes(dataset_name, args.use_config_classes, dataset_config)
        print(f"Using classes for {dataset_name}: {CLASSES}")
        
        # Try to load validation from dedicated folders first
        validation_entries = None
        if args.validation_split > 0:
            print(f"Validation approach: {'Dedicated folders' if not args.force_random_split else 'Random split from training data'}")
            
            if not args.force_random_split:
                validation_entries, val_msg = load_validation_data_from_folders(config_params, dataset_name, CLASSES, args.use_config_classes)
                print(f"Validation status: {val_msg}")
        
        images_dir = args.images_dir if args.images_dir else dataset_config['train_images']
        labels_dir = args.labels_dir if args.labels_dir else dataset_config['train_labels']
        
        print(f"Using dataset: {dataset_name}")
        print(f"Images directory: {images_dir}")
        print(f"Labels directory: {labels_dir}")

    entries = []
    count = 0
    
    # Calculate total sample size for progress reporting
    if not args.use_mixed_dataset:
        img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        n = len(img_files)
        sample_k = int(n * SAMPLE_FRAC)
        print(f"Found {n} images → sampling {sample_k} per class → total ≈ {sample_k*len(CLASSES)} entries")

    for cls in CLASSES:
        system_prompt_text = buil_system_prompt()
        user_prompt_text = build_user_prompt(cls)
        slug = slugify(cls)
        
        # Get class ID for this prompt
        if args.use_mixed_dataset:
            class_id = get_class_id_for_prompt(cls, "se_spill_dataset", None)
            mixed_data = load_mixed_dataset_images(config_params, cls, class_id, sample_k, args.public_data_ratio)
            
            print(f"Processing {len(mixed_data)} mixed samples for class '{cls}'")
            
            for item in mixed_data:
                base = str(count).zfill(len(str(n))+1)
                img_path = os.path.join(item['images_dir'], item['filename'])
                lbl_path = os.path.join(item['labels_dir'], item['filename'][:-4] + ".txt")

                # load image to get dims
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] could not load {item['filename']} from {item['dataset']}, skipping")
                    continue
                h, w = img.shape[:2]

                # parse YOLO labels
                dets = []

                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            label_class_id, xc, yc, wn, hn = map(float, parts)
                            label_class_id = int(label_class_id)
                            
                            # Check if this detection matches what we expect for this prompt
                            if label_class_id == item['source_class_id']:
                                box = yolo_to_coco(xc, yc, wn, hn, w, h)
                                dets.append({"bbox_2d": box, "label": cls})
                else:
                    print(f"[WARN] no label for {item['filename']} from {item['dataset']}, skipping")
                    continue
                
                count += 1
                # build JSON entry with dataset info  
                entry = {
                    "id": f"{slug}_{base}_{item['dataset']}",
                    "source_dataset": item['dataset'],
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nDetect and return the bbox coordinates of the {cls} in the image in COCO format if any."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(dets, indent=2)
                        }
                    ],
                    "image": img_path
                }
                entries.append(entry)
                
        else:
            # Original single dataset logic
            img_files = sorted([
                f for f in os.listdir(images_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))
            ])
            picks = random.sample(img_files, sample_k)

            for fn in picks:
                base = str(count).zfill(len(str(n))+1)
                img_path = os.path.join(images_dir, fn)
                lbl_path = os.path.join(labels_dir, fn[:-4] + ".txt")

                # load image to get dims
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] could not load {fn}, skipping")
                    continue
                h, w = img.shape[:2]

                # parse YOLO labels
                dets = []

                if os.path.exists(lbl_path):
                    with open(lbl_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            class_id, xc, yc, wn, hn = map(float, parts)
                            class_id = int(class_id)
                            
                            # Get the expected class ID for this prompt
                            expected_class_id = get_class_id_for_prompt(cls, dataset_name, dataset_config)
                            
                            # Only include detections that match the expected class for this prompt
                            if class_id == expected_class_id:
                                box = yolo_to_coco(xc, yc, wn, hn, w, h)
                                dets.append({"bbox_2d": box, "label": cls})
                else:
                    print(f"[WARN] no label for {fn}, skipping")
                    continue
                count += 1
                # build JSON entry
                entry = {
                    "id": f"{slug}_{base}",
                    "conversations": [
                        {
                            "from": "human", 
                            "value": f"<image>\nDetect and return the bbox coordinates of the {cls} in the image in COCO format if any."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(dets, indent=2)
                        }
                    ],
                    "image": img_path
                }
                entries.append(entry)

    # Handle validation data
    if args.validation_split > 0:
        if validation_entries is not None:
            # Use validation data from dedicated folders
            print(f"Using validation data from dedicated folders: {len(validation_entries)} examples")
        else:
            # Fall back to random splitting of training data
            print(f"Falling back to random split of training data (validation_split: {args.validation_split:.1%})")
            random.shuffle(entries)
            val_size = int(len(entries) * args.validation_split)
            validation_entries = entries[:val_size]
            entries = entries[val_size:]  # Remove validation examples from training set
        
        # Generate validation output filename
        if args.output.endswith('.json'):
            val_output = args.output[:-5] + '_val.json'
        else:
            val_output = args.output + '_val.json'
        
        # Write validation set
        with open(val_output, 'w') as out:
            json.dump(validation_entries, out, indent=2)
        print(f"Wrote {len(validation_entries)} validation examples to {val_output}")
        
        # Write training set 
        with open(args.output, 'w') as out:
            json.dump(entries, out, indent=2)
        print(f"Wrote {len(entries)} training examples to {args.output}")
        
    else:
        # No validation split - write all as training
        with open(args.output, 'w') as out:
            json.dump(entries, out, indent=2)
        print(f"Wrote {len(entries)} training examples to {args.output}")
    
    # Print summary
    if args.use_mixed_dataset:
        # Count entries by source dataset
        if args.validation_split > 0 and validation_entries is not None:
            primary_count_train = sum(1 for entry in entries if entry.get('source_dataset') == MIXED_DATASET_CONFIG['primary_dataset'])
            secondary_count_train = sum(1 for entry in entries if entry.get('source_dataset') == MIXED_DATASET_CONFIG['secondary_dataset'])
            primary_count_val = sum(1 for entry in validation_entries if entry.get('source_dataset') == MIXED_DATASET_CONFIG['primary_dataset'])
            secondary_count_val = sum(1 for entry in validation_entries if entry.get('source_dataset') == MIXED_DATASET_CONFIG['secondary_dataset'])
            
            print(f"Mixed dataset summary:")
            print(f"  Training set:")
            print(f"    - {MIXED_DATASET_CONFIG['primary_dataset']}: {primary_count_train} examples")
            print(f"    - {MIXED_DATASET_CONFIG['secondary_dataset']}: {secondary_count_train} examples")
            print(f"    - Total: {len(entries)} examples")
            print(f"  Validation set:")
            print(f"    - {MIXED_DATASET_CONFIG['primary_dataset']}: {primary_count_val} examples")
            print(f"    - {MIXED_DATASET_CONFIG['secondary_dataset']}: {secondary_count_val} examples")
            print(f"    - Total: {len(validation_entries)} examples")
        else:
            primary_count = sum(1 for entry in entries if entry.get('source_dataset') == MIXED_DATASET_CONFIG['primary_dataset'])
            secondary_count = sum(1 for entry in entries if entry.get('source_dataset') == MIXED_DATASET_CONFIG['secondary_dataset'])
            print(f"Mixed dataset summary:")
            print(f"  - {MIXED_DATASET_CONFIG['primary_dataset']}: {primary_count} examples")
            print(f"  - {MIXED_DATASET_CONFIG['secondary_dataset']}: {secondary_count} examples")
            print(f"  - Total: {len(entries)} examples")
        
    elif dataset_name == "se_spill_dataset":
        outdoor_water_count = sum(1 for cls in CLASSES if cls == "outdoor-water")
        spill_classes_count = len(CLASSES) - outdoor_water_count
        print(f"Summary for {dataset_name}:")
        print(f"  - outdoor-water: {outdoor_water_count} prompt variation(s)")
        print(f"  - spill-leak-stain: {spill_classes_count} pluralistic prompt variations")
        if args.validation_split > 0 and validation_entries is not None:
            print(f"  - Training examples: {len(entries)}")
            print(f"  - Validation examples: {len(validation_entries)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make few-shot LoRA JSON with configurable dataset support")
    p.add_argument("--config_path", default="config.json",
                   help="Path to the configuration file")
    p.add_argument("--dataset_name", default=None,
                   help="Dataset name to use (spill_dataset or se_spill_dataset). If not specified, uses default from config")
    p.add_argument("--images_dir", default=None,
                   help="Folder with your .jpg/.png images (overrides config)")
    p.add_argument("--labels_dir", default=None,
                   help="Folder with YOLO .txt labels (overrides config)")
    p.add_argument("--use_config_classes", action="store_true",
                   help="Use class mapping from dataset config instead of default classes")
    p.add_argument("--use_mixed_dataset", action="store_true",
                   help="Use mixed dataset training mode (combines se_spill_dataset + spill_dataset)")
    p.add_argument("--public_data_ratio", type=float, default=0.5,
                   help="Scaling factor for public data relative to proprietary data size (e.g., 0.5 = add 50%%, 2.0 = add 200%% of proprietary data size as public data)")
    p.add_argument("--validation_split", type=float, default=0.15,
                   help="Fraction of data to use for validation (e.g., 0.15 = 15%% validation, 85%% training). Set to 0 to disable validation.")
    p.add_argument("--force_random_split", action="store_true",
                   help="Force random splitting of training data instead of using dedicated validation folders")
    p.add_argument("--output",     default="_training_data.json",
                   help="Path to write the JSON array")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for reproducibility")
    args = p.parse_args()
    main(args)
