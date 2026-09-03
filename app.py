import streamlit as st
import whisper
import tempfile
import os
import pandas as pd
import numpy as np
from groq import Groq
import pyttsx3

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
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
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_rag_data():

    chunks = pd.read_pickle(
        "rag_data/chunks_df.pkl"
    )

    embeddings = np.load(
        "rag_data/embeddings.npy"
    )

    return chunks, embeddings


whisper_model = load_whisper_model()
embedding_model = load_embedding_model()

chunks_df, embeddings = load_rag_data()


# --------------------------------------------------
# RAG SEARCH FUNCTION
# --------------------------------------------------

def search_documents(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "Document": chunks_df.iloc[index]["Document"],
            "Similarity": round(
                float(similarities[index]), 4
            ),
            "Text": chunks_df.iloc[index]["Text"]
        })

    return results


# --------------------------------------------------
# OLLAMA RESPONSE FUNCTION
# --------------------------------------------------

def generate_ai_response(question, context):

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = f"""
You are VoiceDesk AI, a helpful customer support assistant.

Answer the user's question using the knowledge base below.

Rules:
- Use only the information in the knowledge base.
- If the answer is present, answer directly.
- Do not say the information is unavailable when it is present.
- Do not invent information.
- Keep the answer short and clear.

Knowledge Base:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
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
# TEXT TO SPEECH FUNCTION
# --------------------------------------------------

def text_to_speech(text):

    audio_path = tempfile.mktemp(
        suffix=".wav"
    )

    engine = pyttsx3.init()

    engine.setProperty(
        "rate",
        160
    )

    engine.say(text)
    engine.save_to_file(
        text,
        audio_path
    )

    engine.runAndWait()
    engine.stop()

    return audio_path


# --------------------------------------------------
# VOICE INPUT
# --------------------------------------------------

st.subheader("🎤 Voice Input")

audio_file = st.audio_input(
    "Speak your question"
)


if audio_file:

    st.success(
        "Voice input received successfully! ✅"
    )

    # Save recorded audio temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(
            audio_file.getvalue()
        )

        audio_path = temp_audio.name


    try:

        # ------------------------------------------
        # WHISPER - SPEECH TO TEXT
        # ------------------------------------------

        with st.spinner(
            "Converting speech to text..."
        ):

            result = whisper_model.transcribe(
                audio_path
            )

        transcribed_text = result[
            "text"
        ].strip()


        if transcribed_text:

            st.subheader(
                "📝 Transcribed Text"
            )

            st.write(
                transcribed_text
            )


            # --------------------------------------
            # RAG - RETRIEVE INFORMATION
            # --------------------------------------

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

                # Use top 3 results as context
                context_parts = []

                for result in results:

                    clean_text = result["Text"]

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


                # ----------------------------------
                # OLLAMA - AI RESPONSE
                # ----------------------------------

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


                # ----------------------------------
                # TEXT TO SPEECH
                # ----------------------------------

                with st.spinner(
                    "Converting response to voice..."
                ):

                    response_audio = text_to_speech(
                        ai_response
                    )


                st.subheader(
                    "🔊 Voice Response"
                )

                st.audio(
                    response_audio,
                    format="audio/wav"
                )


                # ----------------------------------
                # CLEAN TEMP AUDIO
                # ----------------------------------

                if os.path.exists(
                    response_audio
                ):

                    os.remove(
                        response_audio
                    )


            else:

                st.warning(
                    "No relevant information found."
                )


        else:

            st.warning(
                "No speech detected. "
                "Please try again."
            )


    finally:

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )
