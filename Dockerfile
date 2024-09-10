# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port on which the Flask app will run
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]
