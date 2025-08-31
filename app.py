# app.py
# --------------------------------------------------
# 💘 RizzBot – Your AI Wingman (Streamlit front-end)
# --------------------------------------------------
import streamlit as st
import json
from datetime import datetime
import openai
import os
from dotenv import load_dotenv
import re

# ── Setup ──────────────────────────────────────────
load_dotenv()
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="💘 RizzBot", page_icon="💘")
st.title("💘 RizzBot – Rizz up that Baddie!")

# ── Session-state defaults ─────────────────────────
defaults = {
    "rizz_options":        [],
    "selected_rizz":       None,
    "feedback_submitted":  False,
    "last_user_input":     "",
    "generation_idx":      0,      # ← counts how many times "Generate" was pressed
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ── Helper: clean response text ────────────────────
def clean_response(text: str) -> str:
    cleaned = text.strip()

    cleaned = re.sub(r'^(you|me):\s*', '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'^["\']', '', cleaned)
    cleaned = re.sub(r'["\']$', '', cleaned)
    return cleaned.strip()

# ── Helper: remove duplicates while preserving order ──
def remove_duplicates(responses: list[str]) -> list[str]:
    """Remove duplicate responses while preserving order."""
    seen = set()
    unique_responses = []
    for response in responses:
        # Normalize for comparison (lowercase, no extra spaces)
        normalized = response.lower().strip()
        if normalized not in seen and normalized:  # Also skip empty responses
            seen.add(normalized)
            unique_responses.append(response)
    return unique_responses

# ── Helper: generate Rizz lines via OpenAI API ─────
def generate_rizz_candidates(user_input: str, n: int = 1) -> list[str]:
    client = openai.OpenAI()

    prompt = f"Her: {user_input}\nMe:"

    # Generate more than needed to account for duplicates/filtering
    actual_n = min(n * 2, 10) if n > 1 else 1

    resp = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BsALqK92",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are RizzBot, a smooth-talking, bold, confident wingman who helps by giving the user human replies to the input. "
                    "Help the user respond with smooth, witty, confident lines that sound like real texts you'd send someone you're flirting with. Keep it casual — contractions, relaxed tone, no full sentences unless it feels natural. "
                    "Keep them short and punchy — think Tinder pro. "
                    "Respond with just the message text, no prefixes like 'You:' or 'Me:'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,  # Increased for more variety
        top_p=0.9,        # Slightly decreased for better quality
        frequency_penalty=0.5,  # Increased to discourage repetition
        presence_penalty=0.4,   # Increased for more diversity
        max_tokens=100,
        n=actual_n,
    )
    
    # Clean and deduplicate responses
    raw_responses = [c.message.content.strip() for c in resp.choices if c.message and c.message.content]
    cleaned_responses = [clean_response(r) for r in raw_responses if r.strip()]
    unique_responses = remove_duplicates(cleaned_responses)
    
    # Return only the number requested
    return unique_responses[:n] if unique_responses else ["Sorry, couldn't generate a response. Try again!"]

# ── Helper: append interaction to log file ─────────
def log_interaction(user_input, response, feedback, path="rizz_logs.jsonl"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "response":   response,
        "feedback":   feedback,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_positive_example(path="rizz_logs.json"):
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("feedback") == "👍":
                    prompt = f"Her: {entry['user_input']}"
                    response = f"Me: {entry['response']}"
                    examples.append(f"{prompt}\n{response}")

    except FileNotFoundError:
        return []

    return examples

# ── UI: user prompt + options ──────────────────────
user_input = st.text_area("What's the message or situation?", height=150)
smart_mode = st.checkbox("Smart Mode (show multiple options)")

# ── Button: Generate Rizz ──────────────────────────
if st.button("Generate Rizz"):
    if user_input.strip():
        # ask the model
        n_opts = 3 if smart_mode else 1
        st.session_state.rizz_options = generate_rizz_candidates(user_input, n=n_opts)

        # reset per-generation state
        st.session_state.selected_rizz      = None
        st.session_state.feedback_submitted = False
        st.session_state.last_user_input    = user_input

        # advance counter → new widget keys
        st.session_state.generation_idx += 1
    else:
        st.warning("Please enter something first.")

# ── UI: show choices & collect feedback ────────────
if st.session_state.rizz_options:
    # unique keys so widgets start blank each generation
    gen_idx       = st.session_state.generation_idx
    selector_key  = f"rizz_selector_{gen_idx}"
    feedback_key  = f"feedback_radio_{gen_idx}"

    # let user pick the line
    if smart_mode:
        selected = st.radio(
            "Choose your favorite Rizz:",
            st.session_state.rizz_options,
            index=None,
            key=selector_key,
        )
    else:
        selected = st.session_state.rizz_options[0]

    # once a line is chosen, display it and request feedback
    if selected:
        st.session_state.selected_rizz = selected
        st.success(f"💬 RizzBot Suggests:\n\n{selected}")

        if not st.session_state.feedback_submitted:
            feedback = st.radio(
                "Was this good?",
                ["👍", "👎"],
                index=None,
                horizontal=True,
                key=feedback_key,
            )

            # save only after user actively picks 👍 / 👎
            if feedback:
                log_interaction(user_input, selected, feedback)
                st.session_state.feedback_submitted = True
                st.info(f"Feedback saved: {feedback}")