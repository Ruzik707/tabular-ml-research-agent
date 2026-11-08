FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-mvp.txt ./requirements-mvp.txt
RUN python -m pip install --no-cache-dir --requirement requirements-mvp.txt

COPY solution.py README.md .env.example ./
COPY automl_agent ./automl_agent

ENTRYPOINT ["python", "/app/solution.py"]
