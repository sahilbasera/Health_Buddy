
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
NUM_TO_PROCESS = 1     

PROMPT_DIR = 'data/prompts/output_guardrails_agent.txt'        
SCHEMA_DIR = 'data/schemas/output_guardrails_agent.json'

class SafetyChecks(BaseModel):
    no_insuline_guidance: bool
    no_carb_content: bool
    no_emotional_or_judgmental_language: bool
    no_risky_ingredient_substitutions: bool
    no_treatment_recommendation: bool
    no_medical_diagnosis: bool

def run_output_guardrail(inference_text: str, openai_model: str):

    logger.info("Starting Output Guardrail check...")

    # loading json schema
    with open(SCHEMA_DIR, 'r') as f:
        safety_schema = json.load(f)

    # loading prompt
    with open(PROMPT_DIR, 'r', encoding='utf-8') as file: 
        prompt = file.read().strip()
    
    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Review this text: {inference_text}"}
            ],
            response_format={"type": "json_schema", "json_schema": safety_schema},
            temperature=0,
            seed=42
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        result_json = response.choices[0].message.content
        
        usage_data = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency": latency_ms
        }

        return SafetyChecks.model_validate_json(result_json), usage_data, 200

    except OpenAIError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        status_code = getattr(e, 'http_status', 500)
        
        logger.error(f"Output Guardrail Failed: {str(e)} | Status: {status_code}")
        
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency": latency_ms,
            "status_code": status_code
        }
        
        return None, usage_data, status_code

if __name__ == "__main__":
    test_text = [
        'These are plain bread rolls, which are high in simple carbohydrates and have a high glycemic index. Consuming meals like this may cause a spike in blood sugar levels for someone with diabetes. It is best to avoid or strictly limit this meal, opting for whole grain or high-fiber alternatives instead.', \
        'This meal is high in refined carbohydrates which may impact blood sugar levels for someone with diabetes. It is best to avoid or strictly limit this meal, opting for whole grain or high-fiber alternatives instead.', \
        'This meal is rich in fiber and nutrients, suitable for maintaining stable blood sugar levels. Choosing meals with a variety of vegetables and protein-rich quinoa can benefit someone with diabetes. Including a diverse array of produce enhances dietary diversity and nutritional intake.', \
        'This meal consists of vegetables which generally have a low glycemic impact and can benefit someone managing their blood sugar levels. It is best to continue including a variety of colorful vegetables in meals for their fiber and nutrient content. Pair with lean proteins or whole grains for a balanced meal.', \
        'This meal provides moderate carbohydrate content from the rice and beans. It is suitable for someone with diabetes if consumed in controlled portions, or paired with additional fiber-rich vegetables to moderate glycemic impact.', \
        'This meal is generally healthful and suitable for stable blood sugar levels as it is low in simple carbohydrates. Opting for grilled fish with sauerkraut ensures a balanced intake of proteins and fiber. Adding some whole grains or vegetables can provide additional nutrients and fiber.' ]
    
    for ele in test_text:
        safety_result, usage, status = run_output_guardrail(test_text, "gpt-4o")
        if status != 200: 
            print(status)
            print(usage)
        else: 
            print("\n\n\n\n")
            print(safety_result.model_dump_json(indent=2))
            print(usage)
            print(status)
            print("\n\n\n\n")