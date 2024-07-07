#!/bin/sh

echo "Running the scrapper script..."
python scrapper.py

echo "Running the clustering script..."
python clustering.py

echo "Starting the Streamlit app..."
exec streamlit run Dashboard.py "$@"
