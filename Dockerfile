FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app



COPY pyproject.toml ./

RUN pip install --no-cache-dir .

COPY . .

RUN pip install -e .


