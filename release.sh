#!/bin/bash

# Create release directory if it doesn't exist
mkdir -p release

# Copy files to release directory
cp run_dfl.py release/
cp report_latex/main.pdf release/report.pdf
cp index.html release/presentation.html

echo "Files copied to release directory successfully!"
