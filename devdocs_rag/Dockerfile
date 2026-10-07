FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index at image-build time so the container starts ready to serve.
# (For a real production setup with a large/changing corpus, you'd instead
# mount a persistent volume for data/index and run build_index.py as a
# separate one-off job, not on every container build.)
RUN python3 src/build_index.py

EXPOSE 8000
ENV GROQ_API_KEY=""

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
