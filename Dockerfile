FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
COPY pyproject.toml pyproject.toml

RUN python -m pip install --upgrade pip     && python -m pip install -r requirements.txt     && python -m pip install -r requirements-dev.txt     && python -m pip install -e .

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
