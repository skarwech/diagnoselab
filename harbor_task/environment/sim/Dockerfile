FROM python:3.12-slim
# numpy pinned to the repo lockfile: plant hashes and sensor noise depend on it
RUN pip install --no-cache-dir mcp==2.3.0 numpy==2.5.4
RUN useradd -r -u 10001 sim \
 && install -d -o sim -g sim -m 0700 /var/lib/sim/state \
 && install -d -o root -g root -m 0700 /var/lib/sim/out
# diagnoselab/ and docs/ are synced in by `make sync` (gitignored copies of the repo)
COPY diagnoselab/ /app/diagnoselab/
COPY docs/ /app/docs/
ENV PYTHONPATH=/app
USER sim
# DL_SEED / DL_TIER come from this service's compose environment only; no tool ever returns them
CMD ["python", "-m", "diagnoselab.server.app", "--host", "0.0.0.0", "--port", "8000", "--state", "/var/lib/sim/state"]
