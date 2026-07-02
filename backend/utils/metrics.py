"""
Prometheus Metrics — instrumentation for the migration pipeline.
Exposes counters and histograms scraped by Prometheus at /metrics.
"""

from prometheus_client import Counter, Histogram

# --- Migration request tracking ---
migration_requests_total = Counter(
    "migration_requests_total",
    "Total number of migration requests processed",
    ["status", "source_language"],
)

migration_duration_seconds = Histogram(
    "migration_duration_seconds",
    "End-to-end migration pipeline duration in seconds",
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)

# --- Per-agent timing ---
agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Duration of each agent's execution in seconds",
    ["agent_name"],
)

# --- HuggingFace Inference API calls ---
hf_api_calls_total = Counter(
    "hf_api_calls_total",
    "Total HuggingFace Inference API calls",
    ["model", "status"],
)

# --- RAG retrieval timing ---
rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Hybrid RAG retrieval duration in seconds",
)

# --- Risk and validation score distributions ---
risk_score_histogram = Histogram(
    "risk_score_histogram",
    "Distribution of risk scores assigned to migrations",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

validation_score_histogram = Histogram(
    "validation_score_histogram",
    "Distribution of validation scores (0.0 to 1.0)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# --- Procedural construct flagging (fintech-relevant) ---
procedural_flags_counter = Counter(
    "procedural_flags_counter",
    "Count of procedural constructs flagged for manual review",
    ["flag_type"],
)


def record_migration(
    status: str,
    source_language: str,
    duration_ms: float,
    risk_score: float,
    validation_score: float,
) -> None:
    """Record metrics for a completed migration pipeline run."""
    migration_requests_total.labels(status=status, source_language=source_language).inc()
    migration_duration_seconds.observe(duration_ms / 1000.0)
    risk_score_histogram.observe(risk_score)
    validation_score_histogram.observe(validation_score)


def record_agent(agent_name: str, duration_ms: float) -> None:
    """Record execution duration for a single agent node."""
    agent_duration_seconds.labels(agent_name=agent_name).observe(duration_ms / 1000.0)


def record_hf_api(model: str, status: str, duration_ms: float) -> None:
    """Record a HuggingFace Inference API call outcome."""
    hf_api_calls_total.labels(model=model, status=status).inc()
    rag_retrieval_duration_seconds.observe(duration_ms / 1000.0)


def record_procedural_flag(flag_type: str) -> None:
    """Record a detected procedural construct requiring manual review."""
    # Normalize flag_type to a short label for cardinality control
    short_label = flag_type.split(" ")[0].upper() if flag_type else "UNKNOWN"
    procedural_flags_counter.labels(flag_type=short_label).inc()