FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY docker ./docker
COPY run.py .

ENV RECONSTRUCTION_BACKEND=docker
ENV COLMAP_DOCKER_IMAGE=yeicor/colmap-openmvs:cuda-latest
EXPOSE 8000
CMD ["python", "run.py"]
