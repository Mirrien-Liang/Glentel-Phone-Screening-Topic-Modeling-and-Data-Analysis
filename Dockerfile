# Use an official Python runtime as a parent image
FROM python:3.10.13-bookworm AS build

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the custom mymemory_translated.py into the container
# Note: Adjust the path to where the translate package is installed
COPY ./utils/mymemory_translated.py /usr/local/lib/python3.10/site-packages/translate/providers/mymemory_translated.py

# Copy the current directory contents into the container at /app
COPY . .

# Run serve.py when the container launches
CMD ["python", "app.py"]