FROM ultralytics/yolov5:latest-cpu

WORKDIR /usr/src/app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip

# Copy requirements file and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install curl
RUN apt-get update && apt-get install -y curl

# Download the model file
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

# Copy the rest of the application files
COPY . .

# Copy the scripts and make them executable
COPY get_current_region.sh /usr/src/app/get_current_region.sh
COPY start.sh /usr/src/app/start.sh
RUN chmod +x /usr/src/app/get_current_region.sh /usr/src/app/start.sh


# Run the start script
CMD ["/usr/src/app/start.sh"]
