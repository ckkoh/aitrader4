# Multi-stage Dockerfile for AI Trader 4 - Balanced Adaptive Strategy
# Optimized for production deployment with minimal image size

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash trader && \
    mkdir -p /app /app/logs /app/data /app/models && \
    chown -R trader:trader /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=trader:trader . .

# Switch to non-root user
USER trader

# Create necessary directories
RUN mkdir -p logs data models balanced_model_results ensemble_results

# Environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1
ENV OANDA_ENVIRONMENT=practice
ENV LOG_LEVEL=INFO

# Health check (checks if process is running)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; import psutil; exit(0 if any('monitoring_integration' in p.name() or 'streamlit' in p.cmdline() for p in psutil.process_iter(['name', 'cmdline'])) else 1)" || exit 1

# Expose ports
# 8501: Streamlit dashboard
# 8502: Monitoring API (if implemented)
EXPOSE 8501 8502

# Default command: Run monitoring bot
CMD ["python", "monitoring_integration.py"]
