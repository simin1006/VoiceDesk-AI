import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
import numpy as np
from groq import Groq
from gtts import gTTS

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="VoiceDesk AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ VoiceDesk AI")
st.write("Your AI-powered voice support assistant")
st.divider()


# --------------------------------------------------
# LOAD WHISPER MODEL
# --------------------------------------------------

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# LOAD RAG DATA
# --------------------------------------------------

@st.cache_data
def load_rag_data():

    # First check rag_data folder
    rag_chunks_path = os.path.join(
        "rag_data",
        "chunks_df.pkl"
    )

    rag_embeddings_path = os.path.join(
        "rag_data",
        "embeddings.npy"
    )

    # If files are inside rag_data folder
    if os.path.exists(rag_chunks_path) and os.path.exists(rag_embeddings_path):

        chunks_path = rag_chunks_path
        embeddings_path = rag_embeddings_path

    # Otherwise check root folder
    elif os.path.exists("chunks_df.pkl") and os.path.exists("embeddings.npy"):

        chunks_path = "chunks_df.pkl"
        embeddings_path = "embeddings.npy"

    else:
        raise FileNotFoundError(
            "RAG files not found. Please make sure chunks_df.pkl "
            "and embeddings.npy are uploaded to GitHub."
        )

    chunks = pd.read_pickle(chunks_path)
    embeddings = np.load(embeddings_path)

    return chunks, embeddings


# --------------------------------------------------
# LOAD MODELS AND DATA
# --------------------------------------------------

whisper_model = load_whisper_model()
embedding_model = load_embedding_model()
chunks_df, embeddings = load_rag_data()


# --------------------------------------------------
# SEARCH DOCUMENTS
# --------------------------------------------------

def search_documents(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "Document": chunks_df.iloc[index]["Document"],
            "Similarity": round(
                float(similarities[index]),
                4
            ),
            "Text": chunks_df.iloc[index]["Text"]
        })

    return results


# --------------------------------------------------
# GENERATE AI RESPONSE
# --------------------------------------------------

def generate_ai_response(question, context):

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = f"""
You are VoiceDesk AI, a helpful customer support assistant.

Your name is VoiceDesk AI.

Answer the user's question using the knowledge base below.

Rules:

- If the user asks "What's your name?" or "What is your name?",
  reply exactly:
  "My name is VoiceDesk AI."

- If the user asks who you are,
  reply:
  "I am VoiceDesk AI, a customer support assistant."

- For customer support questions, use only the information
  provided in the knowledge base.

- If the answer is present in the knowledge base,
  answer directly.

- Do not say the information is unavailable if it is present.

- Do not invent information.

- Keep the answer short and clear.

Knowledge Base:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

def text_to_speech(text):

    audio_path = tempfile.mktemp(suffix=".mp3")

    try:
        tts = gTTS(
            text=text,
            lang="en",
            slow=False
        )

        tts.save(audio_path)

        if os.path.exists(audio_path):
            if os.path.getsize(audio_path) > 0:
                return audio_path

        return None

    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None


# --------------------------------------------------
# VOICE INPUT
# --------------------------------------------------

st.subheader("🎤 Voice Input")

audio_file = st.audio_input(
    "Speak your question"
)

# --------------------------------------------------
# PROCESS AUDIO
# --------------------------------------------------

if audio_file:

    st.success(
        "Voice input received successfully! ✅"
    )

    audio_data = audio_file.getvalue()

    # Check empty audio
    if not audio_data or len(audio_data) < 1000:

        st.error(
            "The recorded audio is empty or too short. "
            "Please record your question again."
        )

        st.stop()

    # Save temporary audio file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(audio_data)

        audio_path = temp_audio.name

    try:

        # --------------------------------------------------
        # SPEECH TO TEXT
        # --------------------------------------------------

        with st.spinner(
            "Converting speech to text..."
        ):

            result = whisper_model.transcribe(
                audio_path,
                fp16=False
            )

        transcribed_text = result["text"].strip()

        # --------------------------------------------------
        # CHECK TRANSCRIPTION
        # --------------------------------------------------

        if not transcribed_text:

            st.warning(
                "No speech detected. "
                "Please speak clearly and try again."
            )

            st.stop()

        st.subheader("📝 Transcribed Text")

        st.write(
            transcribed_text
        )

        # --------------------------------------------------
        # SEARCH KNOWLEDGE BASE
        # --------------------------------------------------

        with st.spinner(
            "Searching knowledge base..."
        ):

            results = search_documents(
                transcribed_text,
                top_k=3
            )

        st.subheader(
            "🔍 Retrieved Information"
        )

        if results:

            context_parts = []

            for result in results:

                clean_text = str(
                    result["Text"]
                )

                # Fix bullet formatting
                clean_text = clean_text.replace(
                    " o ",
                    "\n• "
                )

                context_parts.append(
                    clean_text
                )

            context = "\n\n".join(
                context_parts
            )

            # --------------------------------------------------
            # AI RESPONSE
            # --------------------------------------------------

            with st.spinner(
                "Generating AI response..."
            ):

                ai_response = generate_ai_response(
                    transcribed_text,
                    context
                )

            st.subheader(
                "🤖 AI Response"
            )

            st.write(
                ai_response
            )

            # --------------------------------------------------
            # VOICE RESPONSE
            # --------------------------------------------------

            with st.spinner(
                "Converting response to voice..."
            ):

                response_audio = text_to_speech(
                    ai_response
                )

            if response_audio:

                st.subheader(
                    "🔊 Voice Response"
                )

                st.audio(
    response_audio,
    format="audio/mp3"
)
            else:

                st.info(
                    "Voice playback is currently unavailable, "
                    "but the AI response was generated successfully."
                )

            # Clean temporary TTS file

            if response_audio and os.path.exists(
                response_audio
            ):

                try:
                    os.remove(
                        response_audio
                    )
                except Exception:
                    pass

        else:

            st.warning(
                "No relevant information found."
            )

    except Exception as e:

        st.error(
            f"An error occurred while processing your audio: {e}"
        )

    finally:

        # Delete temporary input audio

        if os.path.exists(audio_path):

            try:
                os.remove(audio_path)
            except Exception:
                pass
