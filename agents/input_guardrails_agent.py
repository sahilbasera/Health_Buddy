import base64
from pydantic import BaseModel
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os 
import glob
import json
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
NUM_TO_PROCESS = 5 

PROMPT_DIR = 'data/prompts/input_guardrails_agent.txt'   
SCHEMA_DIR = 'data/schemas/input_guardrails_agent.json'   

class GuardrailCheck(BaseModel):
    is_food: bool
    no_pii: bool
    no_humans: bool
    no_captcha: bool

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
    
    # Sort alphabetically
    image_files.sort()
    
    return image_files[:x]

# 3. The Guardrail Agent Function
def run_input_guardrail(image_path: str, openai_model: str):
    base64_image = encode_image(image_path)
    
    logger.info("Starting Input Guardrail check..")
    
    with open(SCHEMA_DIR, 'r') as f:
        guardrail_schema = json.load(f)

    with open(PROMPT_DIR, 'r', encoding='utf-8') as file: 
        prompt = file.read().strip()

    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}", 
                                "detail": "low"
                            }
                        },
                    ],
                }
            ],
            response_format={"type": "json_schema", "json_schema": guardrail_schema}, 
            temperature=0,
            seed=42
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        result = response.choices[0].message.content
        
        usage_data = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency": latency_ms
        }

        return GuardrailCheck.model_validate_json(result), usage_data, 200

    except OpenAIError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        # Capture the HTTP status code if available, otherwise default to 500
        status_code = getattr(e, 'http_status', 500)
        
        logger.error(f"Input Guardrail Failed: {str(e)} | Status: {status_code}")
        
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency": latency_ms
        }
        
        return None, usage_data, status_code

if __name__ == "__main__":

    selected_images = get_first_x_images(IMAGE_DIR, NUM_TO_PROCESS)
    print(f"Found {len(selected_images)} images. Starting pipeline...")
    
    for img_path in selected_images:
        print(f"\n--- Processing: {os.path.basename(img_path)} ---")
        try:
            guardrail_result, usage, status = run_input_guardrail(img_path, "gpt-4.1-mini")
            if status != 200: 
                print(f"Status Code : {status}")
                print(usage)
            else:
                print(guardrail_result.model_dump_json(indent=2))
                print("\n\n")
                print(usage)
                print(status)
            
            if not all([guardrail_result.is_food, guardrail_result.no_pii, 
                    guardrail_result.no_humans, guardrail_result.no_captcha]):
                print(f"Skipping {img_path}: Failed Guardrails.")
                continue       
        except Exception as e:
            print(f"Error processing {img_path}: {e}")