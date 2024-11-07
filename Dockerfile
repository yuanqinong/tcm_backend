# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY ./app ./app

# Expose port 8000
EXPOSE 8000

# Command to run the application with Gunicorn
CMD ["fastapi", "run", "app/main.py", "--port", "8000", "--workers", "4"]