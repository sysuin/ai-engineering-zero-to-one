# Clarity, in a container. Four decisions, each of which has a reason.
FROM python:3.12-slim AS base

# 1. A non-root user. A container that runs as root is a container where a code
#    execution bug becomes a host problem.
RUN useradd --create-home --uid 10001 clarity
WORKDIR /app

# 2. Dependencies in their own layer, before the source. The requirements change
#    rarely and the source changes constantly; this ordering is the difference
#    between a four-second rebuild and a four-minute one.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=clarity:clarity code/ code/
COPY --chown=clarity:clarity data/meridian/index/ data/meridian/index/
USER clarity

# 3. No secrets. Not in an ARG, not in an ENV, not in a file. They arrive at runtime
#    from the platform's secret store, because a secret baked into a layer is in every
#    registry that pulled it and deleting the layer does not remove it.
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/code

# 4. Liveness is the process, not its dependencies — a health check that pings the
#    database restarts the service every time the database hiccups. §26.10.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000
CMD ["uvicorn", "clarity.v1_0.service:app", \
     "--host", "0.0.0.0", "--port", "8000"]
