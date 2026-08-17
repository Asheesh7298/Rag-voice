"""Central config, loaded once from environment. Import `settings` everywhere else."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # STT
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_stt_url: str = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    # Data
    languages: list[str] = os.getenv(
        "LANGUAGES", "as,bn,gu,hi,kn,ml,mr,ne,or,pa,ta,te,ur"
    ).split(",")
    max_passages_per_lang: int = int(os.getenv("MAX_PASSAGES_PER_LANG", "4000"))

    # Embeddings
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))

    # Guardrails
    off_topic_threshold: float = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.35"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.28"))
    groundedness_min_overlap: float = float(os.getenv("GROUNDEDNESS_MIN_OVERLAP", "0.4"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "8"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "20"))

    # Paths
    data_dir: str = "data/processed"
    index_dir: str = "data/index"


settings = Settings()
