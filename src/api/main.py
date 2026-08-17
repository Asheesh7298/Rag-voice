from __future__ import annotations
import logging

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.indexing.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.stt.sarvam_client import SarvamSTTClient
from src.harness.pipeline import Pipeline
from src.harness.schemas import PipelineResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Voice RAG - Indic MSMARCO")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Loaded once at startup, reused across requests -- this is what keeps retrieval fast.
_embed_model = SentenceTransformer(settings.embed_model)
_store = VectorStore.load(settings.index_dir, dim=settings.embed_dim)
_retriever = Retriever(_store, _embed_model)
_llm = LLMClient()
_stt = SarvamSTTClient()
_pipeline = Pipeline(_retriever, _llm, _stt)


@app.get("/health")
def health():
    return {"status": "ok", "index_size": len(_store.metadata)}


@app.post("/query", response_model=PipelineResponse)
def text_query(query: str = Form(...)):
    return _pipeline.run_text_query(query)


@app.post("/voice-query", response_model=PipelineResponse)
async def voice_query(file: UploadFile = File(...), language_code: str | None = Form(None)):
    audio_bytes = await file.read()
    return _pipeline.run_voice_query(audio_bytes, language_code)
