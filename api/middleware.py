import time
from loguru import logger
from fastapi import Request
from prometheus_client import Counter, Histogram

# ── Prometheus Metrics ────────────────────────────────
REQUEST_COUNT = Counter(
    "docmind_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "docmind_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "docmind_errors_total",
    "Total number of errors",
    ["endpoint"]
)


# ── Middleware function ───────────────────────────────
async def metrics_middleware(request: Request, call_next):
    """Track request metrics and log every request."""
    # Record start time BEFORE calling next
    start_time = time.time()
    endpoint = request.url.path

    try:
        # Call the actual endpoint
        response = await call_next(request)
        latency = time.time() - start_time

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint=endpoint
        ).observe(latency)

        # Log request
        logger.info(
            f"{request.method} {endpoint} "
            f"status={response.status_code} "
            f"latency={latency:.3f}s"
        )

        return response

    except Exception as e:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        logger.error(f"Request failed: {endpoint} - {e}")
        raise

# What this means:

# Counter → tracks how many times something happened (total requests, errors)
# Histogram → tracks distributions like latency (p50, p95, p99)
# Every single request automatically gets logged and measured
# Prometheus scrapes these metrics every 15 seconds via /metrics endpoint