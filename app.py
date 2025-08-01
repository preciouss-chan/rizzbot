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
import asyncio
import numpy as np
import pickle
from pathlib import Path

# ── Setup ──────────────────────────────────────────
load_dotenv()
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

if "openai_client" not in st.session_state:
    st.session_state.openai_client = openai.OpenAI()
client = st.session_state.openai_client

if "async_client" not in st.session_state:
    st.session_state.async_client = openai.AsyncOpenAI()
async_client = st.session_state.async_client

st.set_page_config(page_title="💘 RizzBot", page_icon="💘")
st.title("💘 RizzBot – Rizz up that Baddie!")

# ── Session-state defaults ─────────────────────────
defaults = {
    "rizz_options":        [],
    "selected_rizz":       None,
    "feedback_submitted":  False,
    "last_user_input":     "",
    "generation_idx":      0,
    "best_practice_vector": None,  # Cache the vector
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Try to load cached best practice vector at startup
if st.session_state.best_practice_vector is None and Path("best_practice_vector.pkl").exists():
    try:
        with open("best_practice_vector.pkl", "rb") as f:
            st.session_state.best_practice_vector = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load cached best practice vector: {e}")

# ── Helper: Load and process your dataset ──────────
def load_dataset_examples(dataset_path: str) -> list[str]:
    """
    Load examples from your dataset file.
    Supports your custom format: a list of dicts with a "messages" key.
    """
    examples = []
    try:
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Your format: list of dicts with "messages"
                if isinstance(data, list) and data and "messages" in data[0]:
                    for convo in data:
                        for msg in convo.get("messages", []):
                            if msg.get("role") == "assistant":
                                content = msg.get("content", "").strip()
                                if content:
                                    examples.append(content)
                # Add other formats here if needed
        # Option 2: If your dataset is a text file (one example per line)
        elif dataset_path.endswith('.txt'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                examples = [line.strip() for line in f if line.strip()]
        
        # Option 3: If your dataset is a CSV
        elif dataset_path.endswith('.csv'):
            import csv
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Adjust column name based on your CSV structure
                examples = [row['response'] for row in reader if row.get('response')]
        
        # Option 4: If it's a JSONL file (one JSON object per line)
        elif dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Adjust based on your JSONL structure
                        examples.append(entry.get('response', ''))
    
    except FileNotFoundError:
        st.error(f"Dataset file not found: {dataset_path}")
        return []
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []
    
    # Filter out empty examples and ensure quality
    examples = [ex.strip() for ex in examples if isinstance(ex, str) and ex.strip()]
    st.info(f"Loaded {len(examples)} examples from dataset")
    return examples

# ── Helper: Create best practice vector ────────────
async def create_best_practice_vector(examples: list[str], cache_path: str = "best_practice_vector.pkl") -> np.ndarray:
    """
    Create the golden standard vector from your curated examples.
    This can be slow, so we cache the result.
    """
    
    # Check if we have a cached version
    if Path(cache_path).exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_vector = pickle.load(f)
                st.info("Loaded cached best practice vector")
                return cached_vector
        except:
            st.warning("Cache corrupted, regenerating...")
    
    if not examples:
        st.error("No examples provided for best practice vector")
        return np.random.rand(1536)  # Fallback to random
    
    st.info(f"Creating best practice vector from {len(examples)} examples...")
    
    # Get embeddings for all examples
    embedding_tasks = []
    batch_size = 20  # Process in batches to avoid rate limits
    
    all_embeddings = []
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        
        try:
            response = await async_client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [emb.embedding for emb in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Show progress
            st.progress((i + len(batch)) / len(examples))
            
        except Exception as e:
            st.error(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
            continue
    
    if not all_embeddings:
        st.error("Failed to get any embeddings")
        return np.random.rand(1536)
    
    # Calculate the average embedding (centroid)
    embeddings_matrix = np.array(all_embeddings)
    best_practice_vector = np.mean(embeddings_matrix, axis=0)
    
    # Optional: You could also experiment with other methods like:
    # - Weighted average (give more weight to your top examples)
    # - Median instead of mean (more robust to outliers)
    # - Principal component analysis (PCA) to find the main direction
    
    # Cache the result
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(best_practice_vector, f)
        st.success("Best practice vector cached for future use")
    except Exception as e:
        st.warning(f"Could not cache vector: {e}")
    
    return best_practice_vector

# ── Helper: Get best practice vector (main function) ──
async def get_best_practice_vector() -> np.ndarray:
    """
    Get the best practice vector, either from cache or by creating it.
    """
    
    # Check if we already have it in session state
    if st.session_state.best_practice_vector is not None:
        return st.session_state.best_practice_vector
    
    # Path to your dataset - CHANGE THIS to match your file
    dataset_path = "rizz_example.json"  # Change this!
    
    # Load your curated examples
    examples = load_dataset_examples(dataset_path)
    
    if not examples:
        st.error("No examples loaded. Using fallback method...")
        # Fallback: try to use positive examples from logs
        examples = get_positive_examples_from_logs()
    
    if not examples:
        st.error("No good examples found anywhere. Vector will be random.")
        return np.random.rand(1536)
    
    # Create the vector
    vector = await create_best_practice_vector(examples)
    
    # Cache in session state
    st.session_state.best_practice_vector = vector
    
    return vector

# ── Helper: Fallback to logs if dataset not available ──
def get_positive_examples_from_logs(path="rizz_logs.jsonl") -> list[str]:
    """
    Fallback: get positive examples from user feedback logs.
    """
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("feedback") == "👍":
                    examples.append(entry.get("response", ""))
    except FileNotFoundError:
        return []
    
    return [ex for ex in examples if ex.strip()]

# ── Helper: cosine similarity calculation ─────────────
def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ── Helper: async streaming generation ─────────────
async def generate_rizz_streaming(user_input: str) -> str:
    """Generate a single rizz line with streaming for low latency"""
    prompt = f"Her: {user_input}\nMe:"
    
    full_response = ""
    
    stream = await async_client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BsALqK92",
        messages=[
            {
                "role": "user",
                "content": (
                    """
                    You are RizzBot, an AI wingman that helps users craft the perfect response to impress a baddie.
                    Your task is to generate creative, flirty, and engaging responses based on the user's input.
                    The user will provide a message or situation, and you will respond with a line that is
                    charming, witty, and appropriate for the context.
                    Always keep the tone light-hearted and fun, aiming to make a positive impression.   

                    GOAL  
                    • Craft one-liner replies that make her pause and think "smooth."  
                    • Tone = playful confidence, never thirsty, never scripted.

                    HARD RULES  (❌ break = rewrite)  
                    1. No more than 45 characters (including spaces).  
                    2. No generic compliments: beautiful, gorgeous, cute, babe.  
                    3. No pickup-line clichés ("Are you from …", "heaven", "legs tired").  
                    4. Never mention AI, bots, or "this chat".  
                    5. Avoid emojis unless one fits naturally (😉 ok, 😂 usually cringe).

                    STYLE HINTS  (✅ aim for 2-3 per line)  
                    • Light tease or unexpected twist: "Plots ↗ beats  Pics ↘."  
                    • End 20% of lines with a hook question.  
                    • Use contractions & rhythm: "I'm", "can't", "shouldn't".  
                    • When in doubt: mystery > compliment > joke.
                    """
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        max_tokens=100,
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
    
    return full_response.strip()

# ── Helper: temperature sweep with embeddings rerank ──
async def generate_rizz_with_temperature_sweep(user_input: str, n_variants: int = 6) -> list[str]:
    """Generate variants at different temperatures and rerank by similarity to best practices"""
    prompt = f"Her: {user_input}\nMe:"
    
    from openai.types.chat import ChatCompletionUserMessageParam

    base_messages = [
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                """
                You are RizzBot, an AI wingman that helps users craft the perfect response to impress a baddie.
                Your task is to generate creative, flirty, and engaging responses based on the user's input.
                The user will provide a message or situation, and you will respond with a line that is
                charming, witty, and appropriate for the context.
                Always keep the tone light-hearted and fun, aiming to make a positive impression.   

                GOAL  
                • Craft one-liner replies that make her pause and think "smooth."  
                • Tone = playful confidence, never thirsty, never scripted.

                HARD RULES  (❌ break = rewrite)  
                1. No more than 45 characters (including spaces).  
                2. No generic compliments: beautiful, gorgeous, cute, babe.  
                3. No pickup-line clichés ("Are you from …", "heaven", "legs tired").  
                4. Never mention AI, bots, or "this chat".  
                5. Avoid emojis unless one fits naturally (😉 ok, 😂 usually cringe).

                STYLE HINTS  (✅ aim for 2-3 per line)  
                • Light tease or unexpected twist: "Plots ↗ beats  Pics ↘."  
                • End 20% of lines with a hook question.  
                • Use contractions & rhythm: "I'm", "can't", "shouldn't".  
                • When in doubt: mystery > compliment > joke.
                """
            ),
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt
        ),
    ]

    # Generate variants at two different temperatures
    temperatures = [0.5, 0.9]
    tasks = []
    
    for temp in temperatures:
        for _ in range(n_variants // 2):
            task = async_client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:personal::BsALqK92",
                messages=base_messages,
                temperature=temp,
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.3,
                max_tokens=100,
                n=1,
            )
            tasks.append(task)
    
    # Execute all requests concurrently
    responses = await asyncio.gather(*tasks)
    
    # Extract all variants
    variants = []
    for resp in responses:
        variants.append(resp.choices[0].message.content.strip())
    
    # Get embeddings for all variants
    embedding_tasks = []
    for variant in variants:
        task = async_client.embeddings.create(
            model="text-embedding-ada-002",
            input=variant
        )
        embedding_tasks.append(task)
    
    variant_embeddings = await asyncio.gather(*embedding_tasks)
    variant_vectors = [emb.data[0].embedding for emb in variant_embeddings]
    
    # Get the REAL best practice vector
    best_practice_vector = await get_best_practice_vector()
    
    # Calculate similarities and rank
    similarities = []
    for i, variant_vector in enumerate(variant_vectors):
        similarity = calculate_cosine_similarity(variant_vector, best_practice_vector)
        similarities.append((similarity, variants[i]))
    
    # Sort by similarity (highest first) and return top 3
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [variant for _, variant in similarities[:3]]

# ── Helper: generate Rizz lines (fallback sync method) ─────
def generate_rizz_candidates(user_input: str, n: int = 1) -> list[str]:
    """Fallback synchronous generation for non-smart mode"""
    prompt = f"Her: {user_input}\nMe:"

    resp = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BsALqK92",
        messages=[
            {
                "role": "user",
                "content": (
                    """
                    You are RizzBot, an AI wingman that helps users craft the perfect response to impress a baddie.
                    Your task is to generate creative, flirty, and engaging responses based on the user's input.
                    The user will provide a message or situation, and you will respond with a line that is
                    charming, witty, and appropriate for the context.
                    Always keep the tone light-hearted and fun, aiming to make a positive impression.   

                    GOAL  
                    • Craft one-liner replies that make her pause and think "smooth."  
                    • Tone = playful confidence, never thirsty, never scripted.

                    HARD RULES  (❌ break = rewrite)  
                    1. No more than 45 characters (including spaces).  
                    2. No generic compliments: beautiful, gorgeous, cute, babe.  
                    3. No pickup-line clichés ("Are you from …", "heaven", "legs tired").  
                    4. Never mention AI, bots, or "this chat".  
                    5. Avoid emojis unless one fits naturally (😉 ok, 😂 usually cringe).

                    STYLE HINTS  (✅ aim for 2-3 per line)  
                    • Light tease or unexpected twist: "Plots ↗ beats  Pics ↘."  
                    • End 20% of lines with a hook question.  
                    • Use contractions & rhythm: "I'm", "can't", "shouldn't".  
                    • When in doubt: mystery > compliment > joke.
                    """
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        max_tokens=100,
        n=n,
    )
    return [c.message.content.strip() for c in resp.choices] # type: ignore

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

# ── UI: Setup section (optional) ──────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    
    if st.button("🔄 Regenerate Best Practice Vector"):
        # Clear cache and force regeneration
        if Path("best_practice_vector.pkl").exists():
            Path("best_practice_vector.pkl").unlink()
        st.session_state.best_practice_vector = None
        st.success("Cache cleared! Next generation will rebuild the vector.")
    
    # Show current status
    if st.session_state.best_practice_vector is not None:
        st.success("✅ Best practice vector loaded")
    else:
        st.warning("⚠️ Best practice vector not loaded yet")

# ── UI: user prompt + options ──────────────────────
user_input = st.text_area("What's the message or situation?", height=150)
smart_mode = st.checkbox("Smart Mode (show multiple options)")

# ── Button: Generate Rizz ──────────────────────────
if st.button("Generate Rizz"):
    if user_input.strip():
        with st.spinner("Generating your rizz..."):
            if smart_mode:
                # Use temperature sweep with reranking
                try:
                    st.session_state.rizz_options = asyncio.run(
                        generate_rizz_with_temperature_sweep(user_input, n_variants=6)
                    )
                except Exception as e:
                    st.error(f"Smart mode failed: {e}")
                    # Fallback to regular generation
                    st.session_state.rizz_options = generate_rizz_candidates(user_input, n=3)
            else:
                # Use streaming for single response (low latency)
                try:
                    single_response = asyncio.run(generate_rizz_streaming(user_input))
                    st.session_state.rizz_options = [single_response]
                except Exception as e:
                    st.error(f"Streaming failed: {e}")
                    # Fallback to regular generation
                    st.session_state.rizz_options = generate_rizz_candidates(user_input, n=1)

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