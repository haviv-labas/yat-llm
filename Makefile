# Define the Docker image and Dockerfile
DOCKER_IMAGE_NAME=root-micromamba
DOCKER_FILE_PATH=Dockerfile

docker-build:
	@echo "Testing Docker build..."
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f $(DOCKER_FILE_PATH) .

local-pip-install:
	pip install -U numpy==1.23.4 torch==2.0.0 streamlit==1.12.2

run-streamlit-app:
	@echo "Running streamlit app..."
	streamlit run run.py

run-optimisation-via-docker:
	@echo "Optimise the char-level FNN via GP..."
	docker run -v $(PWD):/app $(DOCKER_IMAGE_NAME):latest python -m app.main

check-env:
	env
