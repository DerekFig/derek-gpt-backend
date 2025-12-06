# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed by psycopg2 and friends)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Environment stuff
ENV PYTHONUNBUFFERED=1

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
