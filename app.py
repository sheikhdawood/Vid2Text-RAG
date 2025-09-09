import os
import tempfile
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from better_profanity import profanity
from gtts import gTTS
from transformers import pipeline

# ------------------- Initialization ------------------- #
profanity.load_censor_words()
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file!")
groq_client = Groq(api_key=groq_api_key)

# ------------------- Device ------------------- #
device_whisper = "cpu"  # Always CPU on Apple Silicon
device_embed = "cpu"

# ------------------- Models ------------------- #
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device_embed)
whisper_model = WhisperModel("small", device=device_whisper, compute_type="int8")  # faster on CPU

# Lightweight translation pipelines
translator_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=-1)
translator_ur = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ur", device=-1)

# ------------------- Helper Functions ------------------- #
def translate_text_chunked(text, translator, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append(translator(chunk_text)[0]['translation_text'])
    return " ".join(chunks)

def generate_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    output_path = tempfile.mktemp(suffix=".mp3")
    tts.save(output_path)
    return output_path

def download_audio(youtube_url):
    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "ffmpeg_location": "/opt/homebrew/bin"  # change if different
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    audio_path = os.path.join(tmpdir, "audio.mp3")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    return audio_path

def transcribe_audio(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    transcript = " ".join([seg.text for seg in segments])
    return profanity.censor(transcript)

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(query, chunks, index, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k=top_k)
    return [chunks[i] for i in indices[0]]

def ask_groq(context, question):
    prompt = f"""
You are an assistant answering questions about a YouTube transcript.
Only use the context provided. If unsure, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    raw_answer = response.choices[0].message.content.strip()
    return profanity.censor(raw_answer)

# ------------------- Streamlit App ------------------- #
st.title("🎬 Vid2Text RAG")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Process Video"):
    with st.spinner("Downloading audio..."):
        audio_path = download_audio(youtube_url)
        st.session_state["audio_path"] = audio_path

    with st.spinner("Transcribing audio..."):
        transcript_en = transcribe_audio(audio_path)
        st.session_state["transcript_en"] = transcript_en

    with st.spinner("Chunking & indexing..."):
        chunks = chunk_text(transcript_en)
        index = build_faiss_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = index

    st.success("✅ Processing complete! Transcript ready.")

    # ---------------- Downloads ---------------- #
    st.download_button("📄 Download Transcript (English)", transcript_en, file_name="transcript_en.txt")
    with open(audio_path, "rb") as f:
        st.download_button("🎵 Download Original Audio", f, file_name="audio.mp3", mime="audio/mpeg")

# ---------------- Translation / TTS on-demand ---------------- #
st.subheader("🌐 Translate & Play Audio")
lang_option = st.selectbox("Select Language", ["English", "Hindi", "Urdu"])

if st.button("Translate & Generate Audio"):
    transcript_en = st.session_state.get("transcript_en", "")
    if not transcript_en:
        st.error("Please process a video first!")
    else:
        if lang_option == "English":
            text = transcript_en
            lang_code = "en"
        elif lang_option == "Hindi":
            with st.spinner("Translating to Hindi..."):
                text = translate_text_chunked(transcript_en, translator_hi)
            lang_code = "hi"
        else:
            with st.spinner("Translating to Urdu..."):
                text = translate_text_chunked(transcript_en, translator_ur)
            lang_code = "ur"

        st.write("### Transcript:")
        st.write(text)

        with st.spinner("Generating audio..."):
            audio_file = generate_speech(text, lang=lang_code)
        st.audio(audio_file, format="audio/mp3")
        st.download_button(f"⬇️ Download Transcript Audio ({lang_option})", open(audio_file, "rb"), file_name=f"transcript_{lang_code}.mp3")

# ---------------- Question Answering ---------------- #
question = st.text_input("Ask a question about the video:")

if st.button("Get Answer"):
    if st.session_state.chunks is None:
        st.error("Please process a video first.")
    else:
        with st.spinner("Retrieving answer..."):
            context_chunks = retrieve_chunks(question, st.session_state.chunks, st.session_state.faiss_index)
            context_text = "\n".join(context_chunks)
            answer = ask_groq(context_text, question)

        st.write("**Answer:**", answer)

        st.subheader("🔊 Answer Audio Player")
        audio_file = generate_speech(answer, lang="en")
        st.audio(audio_file, format="audio/mp3")
        st.download_button("⬇️ Download Answer Audio (English)", open(audio_file, "rb"), file_name="answer_en.mp3")
