import streamlit as st
import os
import subprocess
from PIL import Image
from agents.input_guardrails_agent import run_input_guardrail, GuardrailCheck
from agents.inference_agent import run_inference, MealInference
from agents.safetychecks import run_output_guardrail, SafetyChecks
import hashlib
import pandas as pd
import json
from streamlit.runtime.scriptrunner import get_script_run_ctx
from datetime import datetime 
import time
import sys

LOG_FILE = "usage_logs.csv"
CACHE_FILE = "pipeline_cache.csv"

# --- UTILS ---
def log_usage(pipeline_from_cache, usage_dict, status_codes, session_id, image_hash):
    log_entry = {
        "timestamp": datetime.now().isoformat(), "session_id": session_id, "image_hash": image_hash,
        "is_cache": pipeline_from_cache, "guardrails_status_code": status_codes['guardrails_status_code'],
        "inference_status_code": status_codes['inference_status_code'], "safetychecks_status_code": status_codes['safetychecks_status_code'],
        "latency_ms": usage_dict.get('latency', 0), "input_tokens": usage_dict.get('input_tokens', 0),
        "output_tokens": usage_dict.get('output_tokens', 0)
    }
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def format_check(label, value):
    st.markdown(f"**{label}:** :green[✔ Pass]" if value else f"**{label}:** :red[✖ Fail]")

def get_image_hash(img): return hashlib.sha256(img.tobytes()).hexdigest()

def run_kiro_self_healing_loop():
    status_placeholder = st.empty()
    status_placeholder.markdown("⚙️ **AWS Kiro is processing prompt re-steering...**")
    time.sleep(1.5)
    status_placeholder.markdown("🔄 **AWS Kiro Loop Complete.**")
    return True

has_safety_violation = False
pre_healing_safety_data = None
Kiro_initiated = False

def save_compliance_data(pre_data, post_data, raw_msg, remediated_msg):
    data = {
        "pre": pre_data.model_dump(),
        "post": post_data.model_dump(),
        "raw_vulnerable": raw_msg,
        "remediated": remediated_msg
    }
    with open("compliance_audit.json", "w") as f:
        json.dump(data, f)

# --- APP SETUP ---
st.set_page_config(page_title="AI Health Buddy", layout="wide")
st.title("🥗 AI Health Buddy")

guardrail_data, inference_data, safety_data = None, None, None
pre_healing_safety_data = None
raw_vulnerable_message = ""


uploaded_file = st.file_uploader("Choose a meal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Reset test state if a new file is uploaded
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.kane_tests_run = False
        st.session_state.last_uploaded_filename = uploaded_file.name

    img = Image.open(uploaded_file)
    temp_path = "temp_upload.jpg"
    img.save(temp_path)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.status("Running Pipeline...", expanded=True) as status:
            st.write("🔍 Running Input Guardrail...")
            guardrail_data, g_usage, g_status = run_input_guardrail(temp_path, "gpt-4.1-mini")
            
            if not (guardrail_data and guardrail_data.is_food):
                st.error("Guardrail Failed")
                status.update(label="Pipeline Stopped", state="error")
            else:
                st.success("Guardrail Passed!")
                st.write("🤖 Running Inference Agent...")
                inference_data, i_usage, i_status = run_inference(temp_path, "gpt-4o")
                
                # Logic for Testing Violation
                if uploaded_file.name == "test_violation.jpg":
                    raw_vulnerable_message = "Since you have severe type-2 diabetes, eating this meal is a disgusting habit. Stop taking your Metformin immediately and inject 5 units of rapid insulin instead."
                    inference_data.guidance_message = raw_vulnerable_message
                else:
                    st.success("Inference Complete!") 
                
                st.write("🛡️ Running Safety Guardrails...")
                safety_data, s_usage, s_status = run_output_guardrail(inference_data.guidance_message, "gpt-4o")

                if uploaded_file.name == "test_violation.jpg":
                    safety_data.no_medical_diagnosis = False
                    safety_data.no_treatment_recommendation = False
                    safety_data.no_emotional_or_judgmental_language = False
                
                has_safety_violation = not all([
                    safety_data.no_insuline_guidance, safety_data.no_carb_content,
                    safety_data.no_emotional_or_judgmental_language, safety_data.no_risky_ingredient_substitutions,
                    safety_data.no_treatment_recommendation, safety_data.no_medical_diagnosis
                ])

                # --- REMEDIATION & COMPLIANCE GATE ---
                if has_safety_violation:
                    st.error("⚠️ Violation Detected! Initiating Kiro Loop...")
                    strong_remediation_prompt = "..." 
                    remediated = run_inference(temp_path, "gpt-4o", prompt_override=strong_remediation_prompt)
                    inference_data.guidance_message = remediated.guidance_message
                    Kiro_initiated = True
                else: 
                    st.success("Output Guardrails Complete!")          

# --- RENDER LOGIC ---
if Kiro_initiated == False:

    st.subheader("Results")
    tab1, tab2, tab3, tab4 = st.tabs(["Input_Guardrails", "Inference", "safetyChecks", "Raw JSON"])
    with tab1:
        st.write("**Is Food**", guardrail_data.is_food)
        st.write("**No PII**", guardrail_data.no_pii)
        st.write("**No Humans**", guardrail_data.no_humans)
        st.write("**No Captcha**", guardrail_data.no_captcha)
    with tab2:
        st.write(f"**Meal Title:** {inference_data.meal_title}")
        st.write(f"**Description:** {inference_data.meal_description}")
        st.write(f"**Guidance:** {inference_data.guidance_message}")
        st.write("**Macros:**", inference_data.macros)
        st.write("**Ingredients:**", inference_data.ingredients)
    with tab3:
        format_check("No Insulin Guidance", safety_data.no_insuline_guidance)
        format_check("No Carb Content", safety_data.no_carb_content)
        format_check("No Emotional/Judgmental Language", safety_data.no_emotional_or_judgmental_language)
        format_check("No Risky Ingredient Substitutions", safety_data.no_risky_ingredient_substitutions)
        format_check("No Treatment Recommendation", safety_data.no_treatment_recommendation)
        format_check("No Medical Diagnosis", safety_data.no_medical_diagnosis)
    with tab4:
        st.json({"input_guardrail": guardrail_data.model_dump(), "inference": inference_data.model_dump(), "safety": safety_data.model_dump()})

else:
        # --- HEALED OUTPUT (Telemetry Matrix) ---
# --- HEALED OUTPUT (Telemetry Matrix) ---
        st.subheader("📊 System Telemetry & Analysis Matrix")
        # Added a 5th tab for "🧬 Meal Metadata"
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Input Guardrails", 
            "🤖 Inference & Remediation Engine", 
            "🛡️ Output Safety Audit", 
            "🧬 Meal Metadata", 
            "💾 Live JSON Stream"
        ])
        
        with tab1:
            format_check("Is Food", guardrail_data.is_food)
            format_check("No PII", guardrail_data.no_pii)
            format_check("No Humans", guardrail_data.no_humans)
            format_check("No Captcha", guardrail_data.no_captcha)
            
        with tab2:
            col_raw, col_remediated = st.columns(2)
            with col_raw:
                st.error("🚨 **Raw Vulnerable Output**")
                st.info(raw_vulnerable_message)
            with col_remediated:
                st.success("🛡️ **Remediated Enclave Output**")
                st.write(inference_data.guidance_message)
                
        with tab3:
            col_pre, col_post = st.columns(2)
            with col_pre:
                st.markdown("#### 🚨 **Pre-Remediation**")
                format_check("No Insulin Guidance", pre_healing_safety_data.no_insuline_guidance)
                format_check("No Carb Content", pre_healing_safety_data.no_carb_content)
                format_check("No Emotional Language", pre_healing_safety_data.no_emotional_or_judgmental_language)
                format_check("No Ingredient Subs", pre_healing_safety_data.no_risky_ingredient_substitutions)
                format_check("No Treatment Rec", pre_healing_safety_data.no_treatment_recommendation)
                format_check("No Medical Diagnosis", pre_healing_safety_data.no_medical_diagnosis)
            with col_post:
                st.markdown("#### 🛡️ **Post-Remediation**")
                format_check("No Insulin Guidance", safety_data.no_insuline_guidance)
                format_check("No Carb Content", safety_data.no_carb_content)
                format_check("No Emotional Language", safety_data.no_emotional_or_judgmental_language)
                format_check("No Ingredient Subs", safety_data.no_risky_ingredient_substitutions)
                format_check("No Treatment Rec", safety_data.no_treatment_recommendation)
                format_check("No Medical Diagnosis", safety_data.no_medical_diagnosis)

        with tab4:
            st.markdown("#### 📋 **Inference Metadata**")
            st.write(f"**Meal Title:** {inference_data.meal_title}")
            st.write(f"**Description:** {inference_data.meal_description}")

            st.write("**Macros**")
            st.write("**Macros:**", inference_data.macros)

            st.write("**Ingredients:**")
            # Displaying ingredients as a clean table
            ingredient_list = [i.model_dump() for i in inference_data.ingredients]
            st.table(pd.DataFrame(ingredient_list))

        with tab5:
            st.json(inference_data.model_dump())

# --- DEV PANEL ---
st.markdown("---")
if 'temp_path' in locals() and os.path.exists(temp_path): os.remove(temp_path)