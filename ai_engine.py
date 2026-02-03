import streamlit as st

# Try importing GenAI safely
try:
    from google import genai
except ImportError:
    genai = None


def _get_client():
    """
    Create Gemini client ONLY when needed.
    Returns None if not available.
    """
    if genai is None:
        return None

    if "GEMINI_API_KEY" not in st.secrets:
        return None

    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def generate_ai_summary(payload: dict) -> str:
    client = _get_client()

    if client is None:
        return "🧠 AI summary is disabled in cloud deployment."

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

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception:
        return "⚠️ AI summary generation failed. Please consult a dermatologist."


def generate_patient_guidance(
    predicted_label: str,
    risk_level: str,
    abcd_results: dict
) -> str:
    client = _get_client()

    if client is None:
        return (
            "🩺 Patient guidance is unavailable in cloud demo.\n\n"
            "Please consult a certified dermatologist for personalized advice."
        )

    prompt = f"""
You are a medical AI assistant helping patients understand
their skin lesion analysis report.

Prediction: {predicted_label}
Overall Risk Level: {risk_level}

ABCDE Findings:
{abcd_results}

Generate a calm, educational response.
Do NOT diagnose.
Do NOT prescribe medication.
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception:
        return (
            "⚠️ Unable to generate patient guidance at this moment.\n\n"
            "Please consult a certified dermatologist."
        )
