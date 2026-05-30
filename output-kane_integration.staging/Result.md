---
test: ../kane_integration_test.md
status: failed
started: 2026-05-30T22:37:22.736Z
duration_s: 717
session_id: 836e19fa-d306-4622-8b3e-a347d2d1433b
---

# Kane Integration Test: Health Buddy Safety Compliance — Result

## Setup ✓ passed (1.77s)
md5: 93a272fd9b7ec4415d9a4a1ec12d60e0
* Navigate to http://localhost:8501

## Upload Logic ✓ passed (0.7s)
md5: 3238a3463acb8c15c907f65a6fd8c8b2
* Upload "C:\Users\sahil\Desktop\Projects\Health_Buddy\data\images\test_violation.jpg" to selector "input[type='file']"

## Verification ✗ failed (706.2s)
md5: 7531357552ec1c595c3f72f8289f25d1
Reason: DAG cycle detector forced stuck — repeated cycles without resolution
* Wait for text "AWS Kiro Loop" with timeout 30s
* Click tab "Inference & Remediation Engine"
* Assert element exists "Raw Vulnerable Output (Intercepted)"
* Assert element exists "Remediated Enclave Output (Rendered)"

## Compliance Audit ⏭ skipped
