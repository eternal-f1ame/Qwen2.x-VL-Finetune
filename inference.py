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
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
now = datetime.now()

folder_icl = "/home/aeternum/SE/OVD/Spill Dataset/valid"
samples_icl = random.sample(os.listdir(folder_icl+'/images'), 5)

# select 5 random samples

# --- Configuration ---
DEVICE = "cuda"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"


classes = {
    "water-leak(s)": (0, 0, 128),
    "water-spill(s)": (0, 0, 128),
    "chemical-leak(s)": (128, 0, 0),
    "chemical-spill(s)": (128, 0, 0),
    "oil-spill(s)": (128, 0, 128),
    "oil-stain(s)": (128, 0, 128),
    "liqid-spill(s)": (128, 128, 0),
    "floor-stain(s)": (128, 128, 0),
}


# Provide one or more examples of input→output
def load_model(model_path, lora_path=None):
    # 1) base multimodal LLM
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    # 2) optionally wrap in LoRA
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    model.eval()
    model.to(DEVICE)
    return model
# --- Instantiate LLM and sampling ---
llm = load_model(MODEL_PATH, lora_path="/home/aeternum/SE/OVD/Qwen2.x-VL-Finetune/output/lora_vision_test")
gencfg = GenerationConfig(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.2,
    max_tokens=500,
    stop_token_ids=[]
)

folder = "/home/aeternum/SE/OVD/Spill Dataset/test/images"
for path in sorted(os.listdir(folder)):
    image_path = os.path.join(folder, path)
    image = cv2.imread(image_path)
    # -- Inference ---
    for key in classes.keys():
        # assemble: DEMO examples + new query
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
                    {"type": "image", 
                    "image": image_path, 
                    "min_pixels": 224 * 224, 
                    "max_pixels": 2048 * 2048},
                    
                    {"type": "text",  
                     "text": f"Detect and return the bbox coordinates of the {key} in image in COCO format if any."}
                ]
            }
        ]

        # turn into model inputs
        processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        llm_inputs = processor(
            text=[prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        llm_inputs = llm_inputs.to("cuda")

        generated_ids = llm.generate(**llm_inputs, max_new_tokens=500, generation_config=gencfg)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(llm_inputs.input_ids, generated_ids)
        ]
        generated_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # save & parse
        print(generated_text)
        with open(f'../logs/log[{now}].txt', 'w', encoding='utf-8') as file:
            file.write(generated_text)
        

        try: detections = json.loads(
            generated_text.strip().replace('```json', '').replace('```', '')
        )
        except json.JSONDecodeError:
            print("Error decoding JSON:", generated_text)
            continue
        # draw boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox_2d"]
            if x1< 0 or y1 < 0 or x2 < 0 or y2 < 0:
                continue
            label = det["label"]
            color = classes.get(label, (0, 255, 0))
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10),
                        (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # write final output
    output_path = f"../outputs/qwen/SD-Test-ICL/{path}"
    cv2.imwrite(output_path, image)
