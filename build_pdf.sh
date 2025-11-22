#!/bin/bash

# Script to build PDF from LaTeX report
# Usage: ./build_pdf.sh

echo "🔨 Building PDF from LaTeX report..."

# Change to report_latex directory
cd "$(dirname "$0")/report_latex"

# Clean previous build files
echo "🧹 Cleaning previous build files..."
rm -f main.aux main.log main.out main.toc main.bbl main.blg
rm -f chap*/*.aux Chap*/*.aux

# First pass: Generate aux files
echo "📝 First pass: Generating auxiliary files..."
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1 || {
    echo "⚠️  First pass had warnings (continuing...)"
}

# Second pass: Process bibliography (ignore errors if references.bib is empty)
echo "📚 Processing bibliography..."
bibtex main 2>/dev/null || echo "⚠️  No bibliography entries found (skipping)"

# Third pass: Resolve references
echo "🔗 Second pass: Resolving cross-references..."
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || {
    echo "⚠️  Second pass had warnings (continuing...)"
}

# Fourth pass: Final compilation
echo "✨ Final pass: Creating PDF..."
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || {
    echo "⚠️  Final pass had warnings (continuing...)"
}

# Check if PDF was created
if [ -f "main.pdf" ]; then
    PDF_SIZE=$(ls -lh main.pdf | awk '{print $5}')
    PDF_PAGES=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "N/A")
    echo "✅ PDF successfully created!"
    echo "   📄 File: report_latex/main.pdf"
    echo "   📏 Size: $PDF_SIZE"
    echo "   📑 Pages: $PDF_PAGES"
    
    # Open PDF if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🚀 Opening PDF..."
        open main.pdf
    fi
else
    echo "❌ Error: PDF was not created!"
    echo "Check main.log for errors."
    exit 1
fi

echo "🎉 Done!"
