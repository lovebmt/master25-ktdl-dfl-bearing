#!/bin/bash
# Quick PDF build - single command with image sync

# Copy images from reports_dfl to report_latex/image
echo "📁 Copying images from reports_dfl to report_latex/image..."
cp -f reports_dfl/*.png report_latex/image/ 2>/dev/null || echo "⚠️  No PNG files to copy"
cp -f reports_dfl/*.jpg report_latex/image/ 2>/dev/null || true
cp -f reports_dfl/*.json report_latex/image/ 2>/dev/null || true
echo "✅ Images copied"

# Build PDF
cd "$(dirname "$0")/report_latex" && \
pdflatex -interaction=nonstopmode main.tex && \
echo "✅ PDF created: report_latex/main.pdf" && \
open main.pdf 2>/dev/null || echo "📄 PDF ready!"
