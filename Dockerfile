FROM python:alpine3.22

# Set working directory (create it if it doesn't exist)
WORKDIR /code

# Copy requirements.txt
COPY ./requirements.txt /code/requirements.txt

# Install dependencies, --no-cache-dir to avoid caching 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy app files
COPY . /code

# Run the app
CMD ["uvicorn", "API.server:app", "--host", "0.0.0.0", "--port", "8000"]
