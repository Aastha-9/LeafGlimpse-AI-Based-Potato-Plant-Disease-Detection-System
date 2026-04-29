# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY api/requirements.txt ./api/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose the port the app runs on (Hugging Face uses 7860)
EXPOSE 7860

# Run the application
CMD ["sh", "-c", "cd api && uvicorn main:app --host 0.0.0.0 --port 7860"]
