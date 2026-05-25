FROM python:3.11-slim-bullseye

# System dependencies (libgl1 for OpenCV)
RUN apt-get update -qq \
    && apt-get install -y -qq libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# HuggingFace Spaces default port
EXPOSE 7860

CMD ["streamlit", "run", "web_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
