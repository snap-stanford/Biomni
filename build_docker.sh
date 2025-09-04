#!/bin/bash

# Build script for Biomni Docker image
set -e

echo "Building Biomni Docker image..."

# Set image name and tag
IMAGE_NAME="biomni"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Build the Docker image
echo "Building image: $FULL_IMAGE_NAME"
docker build -t $FULL_IMAGE_NAME .

echo "Build completed successfully!"
echo "Image: $FULL_IMAGE_NAME"

# Show image size
echo ""
echo "Image details:"
docker images $IMAGE_NAME

echo ""
echo "To run the container:"
echo "docker run -p 3900:3900 -e ANTHROPIC_API_KEY=your_key_here $FULL_IMAGE_NAME"

echo ""
echo "To test the health endpoint:"
echo "curl http://localhost:3900/health"
