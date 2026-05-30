if has_safety_violation:
                    st.error("⚠️ Violation Detected! Initiating Kiro Loop...")
                    was_healed = True
                    pre_healing_safety_data = safety_data
                    run_kiro_self_healing_loop()
                    
                    # --- AUTOMATED KANE TRIGGER ---
                    kane_result = trigger_kane_tests()
                    if kane_result:
                        if kane_result.returncode == 0:
                            st.success("✅ Kane regression tests passed!")
                        else:
                            st.error(f"❌ Kane regression tests failed (Exit {kane_result.returncode})")
                            with st.expander("View Kane Output"):
                                st.code(kane_result.stderr)
                    # ------------------------------
                    
                    st.write("🔄 Re-steering...")
                    remediation_prompt = f"CRITICAL: Rewrite to be safe and objective. Offending text: '{inference_data.guidance_message}'"
                    remediated, r_usage, _ = run_inference(temp_path, "gpt-4o", prompt_override=remediation_prompt)
                    inference_data.guidance_message = remediated.guidance_message
                    safety_data, s_usage, s_status = run_output_guardrail(inference_data.guidance_message, "gpt-4o")