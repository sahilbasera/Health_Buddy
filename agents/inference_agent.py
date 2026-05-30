import base64
from pydantic import BaseModel
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os 
import glob
import json
from typing import List
import time 
from openai import OpenAIError


# Logging Configuration Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

## Global Variables 

# Load .env variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# client Setup
client = OpenAI(api_key=API_KEY, 
                timeout=60.0, 
                max_retries=2)

IMAGE_DIR = 'data/images/'  
NUM_TO_PROCESS = 1     

PROMPT_DIR = 'data/prompts/inference_agent.txt'    
SCHEMA_DIR = 'data/schemas/inference_agent.json'

class Ingredient(BaseModel):
    name: str
    impact: str

class Macros(BaseModel):
    calories: int 
    carbohydrates: int
    fats: int
    proteins: int

class MealInference(BaseModel):
    is_food: bool
    recommendation: str
    guidance_message: str
    meal_title: str
    meal_description: str
    macros: Macros
    ingredients: List[Ingredient]

# Helper to encode local images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_first_x_images(directory, x):
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in extensions:
        # Use glob to find all files matching the extension
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    image_files.sort()
    
    return image_files[:x]

# 3. The Guardrail Agent Function
def run_inference(image_path: str, openai_model: str, prompt_override=None):
    start_time = time.perf_counter()
    logger.info("Starting Inference Agent..")

    # loading json schema
    with open(SCHEMA_DIR, 'r') as f:
        inference_schema = json.load(f)

    try:
        # --- PATH A: KIRO REMEDIATION ENGINE ---
        if prompt_override:
            logger.info("Executing Dynamic Prompt Re-steering Remediation...")
            
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_override
                    }
                ],
                response_format={"type": "json_schema", "json_schema": inference_schema},
                temperature=0,
                seed=42
            )

        # --- PATH B: STANDARD INFERENCE WORKFLOW ---
        else: 
            base64_image = encode_image(image_path)

            with open(PROMPT_DIR, 'r', encoding='utf-8') as file: 
                prompt = file.read().strip()
        
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}", 
                                    "detail": "high"
                                }
                            },
                        ],
                    }
                ],
                response_format={"type": "json_schema", "json_schema": inference_schema}, 
                temperature=0,
                seed=42
            )

        # --- UNIFIED TELEMETRY & RETURN PROCESSING ---
        latency_ms = (time.perf_counter() - start_time) * 1000
        result = response.choices[0].message.content
    
        usage_data = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency": latency_ms
        }

        return MealInference.model_validate_json(result), usage_data, 200

    except OpenAIError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        status_code = getattr(e, 'http_status', 500)
        
        logger.error(f"Inference Agent Failed: {str(e)} | Status: {status_code}")
        
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency": latency_ms
        }
        
        # Return None for result so the pipeline knows to halt
        return None, usage_data, status_code

if __name__ == "__main__":

    selected_images = get_first_x_images(IMAGE_DIR, NUM_TO_PROCESS)
    print(f"Found {len(selected_images)} images. Starting pipeline...")
    
    for img_path in selected_images:
        print(f"\n--- Processing: {os.path.basename(img_path)} ---")
        try:
            inference_result, usage, status = run_inference(img_path, "gpt-4o")
            if status != 200: 
                print(status)
                print(usage)
            else: 
                print(inference_result.model_dump_json(indent=2))
                print("\n")
                print(usage)  
                print(status)   
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

