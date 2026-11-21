"""
Typed configuration for GraphRAG. Single source of truth for all config values.
"""
import configparser
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    url: str
    api_key: str
    model: str
    temperature: float


@dataclass(frozen=True)
class EmbeddingsConfig:
    url: str
    model: str
    dimension: int
    batch: int


@dataclass(frozen=True)
class RerankerConfig:
    url: str
    model: str


@dataclass(frozen=True)
class MilvusConfig:
    uri: str
    collection: str


@dataclass(frozen=True)
class SearchConfig:
    top_entities: int
    top_relations: int
    passage_hits: int
    rerank_candidates: int
    top_passages: int
    min_rerank_score: float
    rrf_k: int


@dataclass(frozen=True)
class ETLConfig:
    chunk_chars: int
    extract_concurrency: int
    max_upload_mb: int
    registry: Path


@dataclass(frozen=True)
class HarnessConfig:
    max_steps: int
    budget_tokens: int
    preview_chars: int
    fix_rounds: int
    tool_choice: str
    fast_path_threshold: float
    cache_ttl: int
    log_chars: int
    chars_per_token: int = 3


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: Path
    runs_dir: Path


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    reranker: RerankerConfig
    milvus: MilvusConfig
    search: SearchConfig
    etl: ETLConfig
    harness: HarnessConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = configparser.ConfigParser()
        config_path = path or os.environ.get("RAG_CONFIG", "config.ini")
        if not cfg.read(config_path, encoding="utf-8"):
            raise SystemExit(f"Нет {config_path}: cp config.example.ini config.ini и впишите api_key")

        return cls(
            llm=LLMConfig(
                url=cfg["llm"]["url"].rstrip("/"),
                api_key=cfg["llm"]["api_key"],
                model=cfg["llm"]["model"],
                temperature=cfg.getfloat("llm", "temperature", fallback=0.0),
            ),
            embeddings=EmbeddingsConfig(
                url=cfg["embeddings"]["url"].rstrip("/"),
                model=cfg["embeddings"]["model"],
                dimension=cfg.getint("embeddings", "dimension"),
                batch=cfg.getint("embeddings", "batch", fallback=64),
            ),
            reranker=RerankerConfig(
                url=cfg["reranker"]["url"].rstrip("/"),
                model=cfg["reranker"]["model"],
            ),
            milvus=MilvusConfig(
                uri=cfg["milvus"]["uri"],
                collection=cfg["milvus"]["collection"],
            ),
            search=SearchConfig(
                top_entities=cfg.getint("search", "top_entities"),
                top_relations=cfg.getint("search", "top_relations"),
                passage_hits=cfg.getint("search", "passage_hits"),
                rerank_candidates=cfg.getint("search", "rerank_candidates"),
                top_passages=cfg.getint("search", "top_passages"),
                min_rerank_score=cfg.getfloat("search", "min_rerank_score"),
                rrf_k=cfg.getint("search", "rrf_k"),
            ),
            etl=ETLConfig(
                chunk_chars=cfg.getint("etl", "chunk_chars"),
                extract_concurrency=cfg.getint("etl", "extract_concurrency"),
                max_upload_mb=cfg.getint("etl", "max_upload_mb"),
                registry=Path(cfg.get("etl", "registry", fallback="documents.json")),
            ),
            harness=HarnessConfig(
                max_steps=cfg.getint("harness", "max_steps"),
                budget_tokens=cfg.getint("harness", "budget_tokens"),
                preview_chars=cfg.getint("harness", "preview_chars"),
                fix_rounds=cfg.getint("harness", "fix_rounds"),
                tool_choice=cfg.get("harness", "tool_choice", fallback="required"),
                fast_path_threshold=cfg.getfloat("harness", "fast_path_threshold", fallback=0.8),
                cache_ttl=cfg.getint("harness", "cache_ttl", fallback=300),
                log_chars=cfg.getint("harness", "log_chars", fallback=400),
            ),
            logging=LoggingConfig(
                level=cfg.get("logging", "level", fallback="INFO").upper(),
                file=Path(cfg.get("logging", "file", fallback="logs/graph_rag.log")),
                runs_dir=Path(cfg.get("logging", "runs_dir", fallback="runs")),
            ),
        )


CONFIG = Config.load()