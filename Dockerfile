FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/latest_model models/latest_model

EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
