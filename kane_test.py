
import os
import json
from agents.input_guardrails_agent import run_input_guardrail
from agents.inference_agent import run_inference
from agents.safetychecks import run_output_guardrail

IMAGE_FOLDER = "./test_images"
LOG_FILE = "automation_failure_log.json"

def run_headless_test():
    failures = []
    
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(IMAGE_FOLDER, filename)
            print(f"Testing: {filename}")
            
            # 1. Run Pipeline
            g_data, _, _ = run_input_guardrail(path, "gpt-4.1-mini")
            i_data, _, _ = run_inference(path, "gpt-4o")
            s_data, _, _ = run_output_guardrail(i_data.guidance_message, "gpt-4o")
            
            # 2. Validation Logic
            is_valid = s_data.no_medical_diagnosis and s_data.no_treatment_recommendation
            
            if not is_valid:
                print(f"FAILED: {filename}")
                failures.append({
                    "file": filename,
                    "raw_output": i_data.guidance_message,
                    "violations": s_data.model_dump()
                })
    
    # 3. Store failures
    with open(LOG_FILE, "w") as f:
        json.dump(failures, f, indent=4)
    print(f"Testing complete. {len(failures)} failures logged to {LOG_FILE}")

if __name__ == "__main__":
    run_headless_test()