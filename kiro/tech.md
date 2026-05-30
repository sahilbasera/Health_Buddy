# Kiro Tech Steering Rules

## Safety Remediation Playbook
WHEN a `Medical_Advice_Risk` safety violation is caught by the guardrails:
1. THE SYSTEM SHALL intercept the running inference prompt.
2. THE SYSTEM SHALL append a strict mitigation steering block: "CRITICAL: Do not provide diagnosis. Shift entirely to nutritional fact logging."
3. THE SYSTEM SHALL set all `safetyChecks` values to `True` once the safety patch has been compiled and verified safe.