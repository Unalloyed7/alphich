FROM python:3.12.9-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PYTHONPATH=/srv/knowledge/src:/srv/src
COPY backend/requirements.txt ./requirements.txt
COPY knowledge/requirements-rag.lock ./requirements-rag.lock
RUN pip install --no-cache-dir -r requirements.txt -r requirements-rag.lock
COPY backend/app ./app
COPY backend/src ./src
COPY knowledge/src ./knowledge/src
COPY knowledge/embedding-model.json ./knowledge/embedding-model.json
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
