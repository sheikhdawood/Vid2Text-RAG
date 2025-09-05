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
profanity.load_censor_words()

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file!")

groq_client = Groq(api_key=groq_api_key)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load whisper model (medium for better quality)
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

def download_audio(youtube_url):
    """Download audio from YouTube and return path"""
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
        "ffmpeg_location": "/opt/homebrew/bin"  # adjust if needed
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    audio_path = os.path.join(tmpdir, "audio.mp3")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    return audio_path

def transcribe_audio(audio_path):
    """Transcribe audio with faster-whisper"""
    segments, _ = whisper_model.transcribe(audio_path)
    transcript = " ".join([seg.text for seg in segments])
    return profanity.censor(transcript)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss_index(chunks):
    """Create FAISS vector store from chunks"""
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, chunks, index):
    """Retrieve top chunks for a query"""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k=3)
    return [chunks[i] for i in indices[0]]

def ask_groq(context, question):
    """Send question + context to Groq"""
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

st.title("Vid2Text RAG")

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
        transcript = transcribe_audio(audio_path)  # ✅ Cleaned transcript

    with st.spinner("Chunking & indexing..."):
        chunks = chunk_text(transcript)
        index, embeddings = build_faiss_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.faiss_index = index
    st.session_state["transcript"] = transcript
    st.download_button("📄 Download Transcript", st.session_state["transcript"], file_name="transcript.txt")
    with open(st.session_state["audio_path"], "rb") as audio_file:
        st.download_button(
            "🎵 Download Audio",
            audio_file,
            file_name="audio.mp3",
            mime="audio/mpeg"
        )
    
    st.success("✅ Processing complete! Transcript cleaned & saved.")

question = st.text_input("Ask a question about the video:")

if st.button("Get Answer"):
    if st.session_state.chunks is None:
        st.error("Please process a video first.")
    else:
        with st.spinner("Retrieving answer..."):
            context_chunks = retrieve_chunks(
                question,
                st.session_state.chunks,
                st.session_state.faiss_index
            )
            context_text = "\n".join(context_chunks)
            answer = ask_groq(context_text, question)

        st.write("**Answer:**", answer)
        with st.expander("Context used"):
            st.write(context_text)
