#!/bin/bash

# Change to the project directory
# cd /workdir_efs/jhjeon/Biomni

# # Create workspace directory if it doesn't exist
# mkdir -p streamlit_workspace

# Run Streamlit app

streamlit run main_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
