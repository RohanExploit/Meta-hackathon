FROM python:3.10-slim

WORKDIR /app

# Copy entire repo (editable install needs all package files)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
