# Use official Python 3.12 slim image as base
FROM python:3.12-slim-bookworm

# Set working directory inside the container
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && apt-get install -y python3-distutils libsqlite3-dev libgl1 libglib2.0-0 tesseract-ocr

# Download and install `uv`
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure `uv` and `uvicorn` are on PATH
ENV PATH="/root/.local/bin:$PATH"

# Create requirements file for uv
RUN echo "numpy\nscipy\nopencv-python\npillow\npytesseract\nMarkdown\nrequests\nbeautifulsoup4\naiofiles\npython-multipart\nFaker\nfastapi\nopenai\nuvicorn" > /tmp/requirements.txt

# Install dependencies using `uv`
RUN uv pip install --system -r /tmp/requirements.txt


RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash && \
    apt-get install -y nodejs && \
    rm -f /usr/bin/npx && \
    npm install -g npx prettier@3.4.2 && \
    rm -rf /var/lib/apt/lists/*


# Copy the FastAPI app files into the container
COPY Functions.py Server.py /app/

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "Server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
