# Lightweight CPU image for the data pipeline and the test suite.
# The repository is bind-mounted at /work at run time (see Makefile).
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/work

# CPU-only torch: the image runs tests and small CPU checks, GPU work runs on rented hosts
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml /tmp/pyproject.toml
RUN python - <<'PY' > /tmp/requirements.txt
import tomllib
p = tomllib.load(open("/tmp/pyproject.toml", "rb"))["project"]
extra = p["optional-dependencies"]
print("\n".join(p["dependencies"] + extra["dev"] + extra["features"]))
PY
RUN pip install -r /tmp/requirements.txt

WORKDIR /work
