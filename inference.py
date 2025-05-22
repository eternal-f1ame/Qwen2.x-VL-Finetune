import os
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
    parser = argparse.ArgumentParser(description="Run inference with Qwen 2.5 VL model.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images and JSON data.")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to LoRA adapter.")
    args = parser.parse_args()

    config_params = load_config(args.config_path)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    output_images_dir = os.path.join(args.output_dir, "images")
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
    # model.to(device) # device_map="auto" handles this

    # GenerationConfig
    gen_cfg_params = {
        "temperature": config_params['temperature'],
        "top_p": config_params['top_p'],
        "repetition_penalty": config_params['repetition_penalty'],
        "max_new_tokens": config_params['max_new_tokens'],
        "stop_token_ids": processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else [] # Common practice
    }
    if config_params['top_k'] is not None:
        gen_cfg_params["top_k"] = config_params['top_k']
    gencfg = GenerationConfig.from_dict(gen_cfg_params)

    # Example classes (adjust as needed or load from config if preferred)
    classes_for_detection = {
        "water-leak(s)": (0, 0, 128),
        "water-spill(s)": (0, 0, 128),
        "chemical-leak(s)": (128, 0, 0),
        # Add other classes you intend to detect
    }

    output_json_data = []
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        
        detections_for_image = {"image_path": image_filename, "detections": []}

        for class_key, color in classes_for_detection.items():
            prompt_text = config_params['prompt'].format(class_key) # Use prompt from config
            
            messages = [
                {
                    'role': 'system',
                    'content': [
                        {"type": "text", 
                         "text": "You are an adept Factory Indoors Leaks, Spills and Object Detection model. Strictly Avoid False Positives."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path, "min_pixels": 224 * 224, "max_pixels": 2048 * 2048},
                        {"type": "text",  "text": prompt_text}
                    ]
                }
            ]

            # Prepare inputs for the model
            try:
                text_prompt_for_tokenizer = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                llm_inputs = processor(
                    text=[text_prompt_for_tokenizer],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device) # Ensure inputs are on the same device as the model
            except Exception as e:
                print(f"Error processing inputs for {image_filename} with class {class_key}: {e}")
                continue

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**llm_inputs, generation_config=gencfg)
            
            generated_ids_trimmed = generated_ids[:, llm_inputs.input_ids.shape[1]:] # Trim input tokens
            generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            log_file_path = os.path.join(log_dir, f"log_{image_filename.split('.')[0]}_{class_key}_{now.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(log_file_path, 'w', encoding='utf-8') as f_log:
                f_log.write(f"Image: {image_filename}, Class: {class_key}\n")
                f_log.write(f"Prompt:\n{prompt_text}\n")
                f_log.write(f"Generated Text (raw):\n{generated_text}\n")

            try:
                parsed_detections = json.loads(generated_text.strip().replace('```json', '').replace('```', ''))
                if isinstance(parsed_detections, list):
                    for det in parsed_detections:
                        if isinstance(det, dict) and "bbox_2d" in det and "label" in det:
                            x1, y1, x2, y2 = map(int, det["bbox_2d"])
                            label = det["label"]
                            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2: # Basic validation
                                print(f"Skipping invalid bbox for {label} in {image_filename}: {[x1,y1,x2,y2]}")
                                continue
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            detections_for_image["detections"].append(det)
                        else:
                            print(f"Warning: Detection item has unexpected format: {det} for {image_filename}")
                else:
                     print(f"Warning: Parsed detection is not a list: {parsed_detections} for {image_filename}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON for {image_filename}, class {class_key}: {generated_text}")
            except Exception as e:
                print(f"Error drawing boxes or processing detections for {image_filename}, class {class_key}: {e}")
        
        output_image_path = os.path.join(output_images_dir, image_filename)
        cv2.imwrite(output_image_path, image)
        output_json_data.append(detections_for_image)

    output_json_file = os.path.join(args.output_dir, "detections_summary.json")
    with open(output_json_file, 'w') as f:
        json.dump(output_json_data, f, indent=4)
    print(f"Processing complete. Output images in {output_images_dir}, summary in {output_json_file}, logs in {log_dir}")

if __name__ == "__main__":
    main()
