import os
import random
import cv2
import json
import torch
from transformers import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    GenerationConfig
)
from peft import PeftModel # If you use LoRA, keep this
from qwen_vl_utils import process_vision_info # Utility for Qwen
from datetime import datetime
import warnings
import argparse
from tqdm import tqdm # Added for progress bar

warnings.filterwarnings('ignore')
now = datetime.now()

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run ICL inference with Qwen 2.5 VL model.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the test images.")
    parser.add_argument("--icl_image_dir", type=str, required=True, help="Directory containing the ICL images.")
    parser.add_argument("--icl_label_dir", type=str, required=True, help="Directory containing the ICL labels (JSON files with detection data).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images and logs.")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to LoRA adapter.")
    parser.add_argument("--num_icl_examples", type=int, default=5, help="Number of ICL examples to use.")
    args = parser.parse_args()

    config_params = load_config(args.config_path)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    output_images_dir = os.path.join(args.output_dir, "images_icl") # Separate output for ICL
    os.makedirs(output_images_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    processor = Qwen2_5_VLProcessor.from_pretrained(config_params['processor_name'], trust_remote_code=True, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config_params['model_name'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16)
    model.eval()

    # GenerationConfig
    gen_cfg_params = {
        "temperature": config_params['temperature'],
        "top_p": config_params['top_p'],
        "repetition_penalty": config_params['repetition_penalty'],
        "max_new_tokens": config_params['max_new_tokens'],
         "stop_token_ids": processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else []
    }
    if config_params['top_k'] is not None:
        gen_cfg_params["top_k"] = config_params['top_k']
    gencfg = GenerationConfig.from_dict(gen_cfg_params)
    
    # Define classes (could also be part of config_params or args)
    classes_for_detection = {
        "water-leak(s)": (0, 0, 128),
        "water-spill(s)": (0, 0, 128),
        "chemical-leak(s)": (128, 0, 0),
        "chemical-spill(s)": (128, 0, 0),
        "oil-spill(s)": (128, 0, 128),
        "oil-stain(s)": (128, 0, 128),
        # Add other classes you expect from ICL examples and want to detect
    }

    # --- Prepare In-context learning examples ---
    icl_image_files = os.listdir(args.icl_image_dir)
    selected_icl_files = random.sample(icl_image_files, min(args.num_icl_examples, len(icl_image_files)))
    
    demo_messages = []
    # System message (optional, but often good for Qwen)
    demo_messages.append({
        "role": "system",
        "content": "You are an adept Factory Indoors Leaks, Spills and Object Detection model. Strictly Avoid False Positives."
    })

    for icl_filename in selected_icl_files:
        icl_image_path = os.path.join(args.icl_image_dir, icl_filename)
        # Try to find a matching label file (e.g., .json or .txt)
        base_name = os.path.splitext(icl_filename)[0]
        icl_label_path_json = os.path.join(args.icl_label_dir, base_name + '.json')
        icl_label_path_txt = os.path.join(args.icl_label_dir, base_name + '.txt') # if using txt for labels
        
        example_detections = []
        example_prompt_key = random.choice(list(classes_for_detection.keys())) # Default/random key if label is generic

        if os.path.exists(icl_label_path_json):
            try:
                with open(icl_label_path_json, 'r', encoding='utf-8') as f_label:
                    label_data = json.load(f_label)
                    # Assuming label_data is a list of detections like [{"bbox_2d": [x1,y1,x2,y2], "label": "class_name"}, ...]
                    # Or it could be structured per image, e.g. {"image_name": ..., "detections": [...]}
                    # Adjust parsing based on your actual ICL label format.
                    if isinstance(label_data, list) and label_data: # Simple list of detections format
                        example_detections = label_data
                        if example_detections[0].get("label"): # Try to get a relevant class for the prompt
                             example_prompt_key = example_detections[0]["label"]
                    elif isinstance(label_data, dict) and "detections" in label_data: # Dict with detections key
                        example_detections = label_data["detections"]
                        if example_detections and example_detections[0].get("label"): 
                            example_prompt_key = example_detections[0]["label"]
            except Exception as e:
                print(f"Error reading or parsing ICL JSON label {icl_label_path_json}: {e}")
        elif os.path.exists(icl_label_path_txt):
             # Add logic here if your .txt files have a parseable format for detections
            print(f"Note: ICL .txt label parsing not fully implemented yet for {icl_label_path_txt}")

        if not example_detections:
            print(f"Warning: No valid detections found for ICL example {icl_filename}. Skipping this example.")
            continue

        # User turn with image + task for ICL example
        demo_messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": icl_image_path, "min_pixels": 224 * 224, "max_pixels": 2048 * 2048},
                {"type": "text",  "text": config_params['prompt'].format(example_prompt_key)}
            ]
        })
        # Assistant turn with the "correct" JSON result for ICL example
        demo_messages.append({
            "role": "assistant",
            "content": "```json\n" + json.dumps(example_detections, indent=2) + "\n```"
        })

    if len(demo_messages) <=1: # Only system prompt
        print("Warning: Not enough valid ICL examples were prepared. Proceeding without ICL or with minimal context.")

    output_json_data = []
    test_image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_filename in tqdm(test_image_files, desc="Processing test images with ICL"):
        current_image_path = os.path.join(args.image_dir, image_filename)
        image_display = cv2.imread(current_image_path) # For drawing boxes later
        if image_display is None:
            print(f"Warning: Could not read test image {current_image_path} with OpenCV. Skipping.")
            continue

        detections_for_image = {"image_path": image_filename, "detections": []}

        # For Qwen, you typically query for one class at a time if using specific prompts.
        # If your ICL examples cover multiple classes, the model might learn to output them together.
        # Here, we will iterate through target classes for detection on the new image.
        for target_class_key, color in classes_for_detection.items():
            query_prompt_text = config_params['prompt'].format(target_class_key)
            
            # Assemble final messages: DEMO examples + new query
            current_query_messages = demo_messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": current_image_path, "min_pixels": 224 * 224, "max_pixels": 2048 * 2048},
                        {"type": "text",  "text": query_prompt_text}
                    ],
                }
            ]
            
            try:
                text_prompt_for_tokenizer = processor.apply_chat_template(current_query_messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(current_query_messages)
                llm_inputs = processor(
                    text=[text_prompt_for_tokenizer],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
            except Exception as e:
                print(f"Error processing inputs for {image_filename} with class {target_class_key} (ICL): {e}")
                continue

            with torch.no_grad():
                generated_ids = model.generate(**llm_inputs, generation_config=gencfg)
            
            generated_ids_trimmed = generated_ids[:, llm_inputs.input_ids.shape[1]:]
            generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            log_file_path = os.path.join(log_dir, f"log_icl_{image_filename.split('.')[0]}_{target_class_key}_{now.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(log_file_path, 'w', encoding='utf-8') as f_log:
                f_log.write(f"Image: {image_filename}, Target Class: {target_class_key}\n")
                f_log.write(f"Prompt (final query part):\n{query_prompt_text}\n")
                f_log.write(f"Generated Text (raw):\n{generated_text}\n")

            try:
                parsed_detections = json.loads(generated_text.strip().replace('```json', '').replace('```', ''))
                if isinstance(parsed_detections, list):
                    for det in parsed_detections:
                        if isinstance(det, dict) and "bbox_2d" in det and "label" in det:
                            # Filter by target_class_key or trust model's label if ICL is strong
                            # if det["label"] == target_class_key: # Optional: stricter filtering
                            x1, y1, x2, y2 = map(int, det["bbox_2d"])
                            label = det["label"] # Use label from model output
                            
                            # Use color of the *detected* label if available, else fallback or use query color
                            current_color = classes_for_detection.get(label, color) 

                            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2:
                                print(f"Skipping invalid bbox for {label} in {image_filename}: {[x1,y1,x2,y2]}")
                                continue
                            cv2.rectangle(image_display, (x1, y1), (x2, y2), current_color, 2)
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(image_display, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), current_color, -1)
                            cv2.putText(image_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            detections_for_image["detections"].append(det)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {image_filename} (ICL), class {target_class_key}: {generated_text}")
            except Exception as e:
                print(f"Error drawing/processing detections for {image_filename} (ICL), class {target_class_key}: {e}")

        output_image_path = os.path.join(output_images_dir, image_filename)
        cv2.imwrite(output_image_path, image_display)
        output_json_data.append(detections_for_image)

    output_json_file = os.path.join(args.output_dir, "detections_summary_icl.json")
    with open(output_json_file, 'w') as f:
        json.dump(output_json_data, f, indent=4)
    print(f"ICL Processing complete. Output images in {output_images_dir}, summary in {output_json_file}, logs in {log_dir}")

if __name__ == "__main__":
    main()
