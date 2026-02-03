try:
    from google import genai
except ImportError:
    genai = None

import streamlit as st

# Create Gemini client (NEW SDK)
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
def generate_ai_summary(text):
    if genai is None:
        return "AI summary disabled in cloud demo."
    
    # existing AI logic

def generate_ai_summary(payload: dict) -> str:
    prompt = f"""
You are a medical AI assistant.

Given the following AI-based skin lesion analysis,
provide:
1. A concise clinical-style summary
2. Risk interpretation
3. Recommended next steps
4. Clear disclaimer

Rules:
- Do NOT diagnose
- Do NOT claim certainty
- Keep language professional and educational

DATA:
{payload}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def generate_patient_guidance(
    predicted_label: str,
    risk_level: str,
    abcd_results: dict
):
    """
    Generates patient-focused guidance based on AI results.
    Output is clear, non-alarming, medical-support language.
    """

    prompt = f"""
You are a medical AI assistant helping patients understand
their skin lesion analysis report.

The goal is to provide:
- Clear explanations
- Practical precautions
- Lifestyle and skin-care guidance
- Calm, supportive tone
- No diagnosis claims

--- PATIENT REPORT DATA ---

Prediction: {predicted_label}
Overall Risk Level: {risk_level}

ABCDE Findings:
{abcd_results}

--- INSTRUCTIONS ---

Generate the response in MARKDOWN format with the following sections:

## 1. What This Result Means
Explain in simple terms what the prediction and risk level indicate.

## 2. Immediate Precautions
List practical steps the patient should follow now.

## 3. Skin Protection & Daily Care
Explain sun protection, clothing, skincare habits.

## 4. Diet & Lifestyle Support
Suggest realistic food and lifestyle choices that support skin health.
Do NOT prescribe medication.

## 5. Self-Monitoring Guidance
Explain how the patient can safely observe changes over time.

## 6. When to Seek Medical Help
Clearly list warning signs that require dermatologist consultation.

## 7. Emotional Reassurance
Provide calm reassurance without minimizing risk.

## 8. Disclaimer
State clearly this is not a diagnosis.

Use bullet points where helpful.
Avoid medical jargon.
Avoid fear-inducing language.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        return (
            "⚠️ Unable to generate patient guidance at this moment.\n\n"
            "Please consult a certified dermatologist for personalized advice."
        )

