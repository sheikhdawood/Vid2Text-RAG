# 🎬 Voxify

**Voxify** is a Streamlit-based application that converts YouTube videos into multilingual transcripts, provides audio playback for each language, and allows question-answering over the video content using Groq AI. It integrates advanced AI models for transcription, translation, embeddings, and TTS.

---

## 🚀 Features

- **YouTube Video Processing**
  - Download audio from YouTube videos.
  - Convert video audio to clean transcripts using Whisper.
  - Profanity filter applied to transcripts.

- **Multilingual Support**
  - Translate transcripts to **Hindi**, **Urdu**, and other languages.
  - Audio playback for all languages using gTTS.
  - Downloadable text and audio files.

- **Chunked Text Indexing**
  - Splits transcripts into chunks.
  - Builds embeddings with Sentence Transformers.
  - Searches relevant chunks for question-answering.

- **Question Answering**
  - Ask questions about video content.
  - AI-powered answers using **Groq LLM**.
  - Context visualization for transparency.

- **Audio Features**
  - Listen to transcripts in multiple languages.
  - Listen to AI-generated answers.

---

## 📦 Tech Stack

- **Frontend:** Streamlit
- **Transcription:** Faster Whisper (`faster-whisper`)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **Translation:** Hugging Face Transformers (`Helsinki-NLP/opus-mt-*`)
- **TTS:** gTTS
- **Profanity Filter:** `better-profanity`
- **LLM QA:** Groq API
- **Environment:** Python 3.9+, .env for API keys

---

## ⚡ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/vid2text-rag.git
cd vid2text-rag
