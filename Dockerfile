FROM python:3.11-slim

WORKDIR /app

# system deps (pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

CMD ["python", "bot.py"]
