import json
import argparse
import os

# System prompt based on inference.py
SYSTEM_PROMPT = "You are an adept Factory Indoors Leaks, Spills and Object Detection model. Strictly Avoid False Positives."

def convert_single_sample_to_role_content(sample_data, image_base_directory=""):
    """
    Converts a single data sample from the "from-value" format to the "role-content" format.

    Args:
        sample_data (dict): A dictionary representing a single sample,
                            containing "image" and "conversations".
        image_base_directory (str, optional): Base path for relative image file paths. Defaults to "".

    Returns:
        list: A list of message dictionaries for the converted sample, or None if conversion fails.
    """
    if "image" not in sample_data or not isinstance(sample_data.get("image"), str):
        print(f"Warning: Missing or invalid 'image_file' in sample: {sample_data}. Skipping sample.")
        return None
    if "conversations" not in sample_data or not isinstance(sample_data.get("conversations"), list):
        print(f"Warning: Missing or invalid 'conversations' in sample: {sample_data}. Skipping sample.")
        return None

    image_file_path = sample_data["image"]
    if image_base_directory and not os.path.isabs(image_file_path):
        image_file_path = os.path.join(image_base_directory, image_file_path)

    # Check if image file exists, if a base directory is provided (implying local access is expected)
    # For full paths, we assume they are accessible where training will occur.
    if image_base_directory and not os.path.isfile(image_file_path):
        print(f"Warning: Image file not found at resolved path: {image_file_path} (original: {sample_data['image_file']}). Proceeding, but ensure path is correct for training.")


    messages = []
    # Add system prompt
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    })

    first_user_turn = True
    for turn_index, turn in enumerate(sample_data["conversations"]):
        if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
            print(f"Warning: Invalid turn format in sample {sample_data.get('id', 'Unknown_ID')}, turn {turn_index}. Skipping this turn: {turn}")
            continue

        source_role = turn["from"].lower()
        value = turn["value"]
        role_to_set = None

        if source_role in ["human", "user"]:
            role_to_set = "user"
        elif source_role in ["gpt", "assistant"]:
            role_to_set = "assistant"
        else:
            print(f"Warning: Unknown role '{turn['from']}' in sample. Skipping this turn.")
            continue

        current_content = []
        if role_to_set == "user":
            if first_user_turn:
                # Add image only to the first user turn of the conversation
                current_content.append({"type": "image", "image": image_file_path})
                first_user_turn = False
            current_content.append({"type": "text", "text": value})
            messages.append({"role": "user", "content": current_content})
        elif role_to_set == "assistant":
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": value}]
            })
    
    if not any(msg['role'] == 'user' for msg in messages):
        print(f"Warning: No user turns found or processed for sample originally with image {sample_data['image_file']}. This might lead to an invalid training sample.")
        return None # Or return messages if partial processing is acceptable

    return messages

def main():
    parser = argparse.ArgumentParser(description="Convert training data from 'from-value' to 'role-content' format for VLM.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSON file (e.g., data in 'from-value' format).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the converted JSON file (in 'role-content' format).")
    parser.add_argument("--image_base_dir", type=str, default="",
                        help="Optional base directory for relative image paths in the input file. "
                             "If provided, relative image paths will be joined with this directory.")
    
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {args.input_file}. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the input file: {e}")
        return

    if not isinstance(input_data, list):
        print("Error: Input JSON data must be a list of samples.")
        return

    converted_data_all_samples = []
    successful_conversions = 0
    failed_conversions = 0

    for i, sample in enumerate(input_data):
        if not isinstance(sample, dict):
            print(f"Warning: Item {i} in the input list is not a dictionary. Skipping.")
            failed_conversions +=1
            continue
        
        # Assuming each sample might have an 'id' for better logging
        sample_id_info = f"sample index {i}"
        if 'id' in sample :
            sample_id_info = f"sample with id '{sample['id']}' (index {i})"

        converted_sample_messages = convert_single_sample_to_role_content(sample, args.image_base_dir)
        if converted_sample_messages:
            converted_data_all_samples.append(converted_sample_messages)
            successful_conversions +=1
        else:
            print(f"Failed to convert {sample_id_info}.")
            failed_conversions +=1


    if not converted_data_all_samples:
        print("No data was successfully converted. Output file will not be created.")
        return

    try:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True) # Ensure output directory exists
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data_all_samples, f, indent=2)
        print(f"\nConversion complete.")
        print(f"Successfully converted {successful_conversions} samples.")
        if failed_conversions > 0:
            print(f"Failed to convert {failed_conversions} samples (see warnings above).")
        print(f"Output saved to {args.output_file}")
    except IOError as e:
        print(f"Error: Could not write to output file {args.output_file}. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}")

if __name__ == "__main__":
    main() 