# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Flask app to the working directory
COPY . .

# Expose port 5001 (the port Flask will run on)
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask app with Gunicorn (for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
