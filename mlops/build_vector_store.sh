#!/bin/bash
set -e

# vars
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
VECTOR_STORE_DIR="$PROJECT_ROOT/vector_store"

echo "=== Building Vector Store ==="
echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo "Vector store directory: $VECTOR_STORE_DIR"

if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory..."
    mkdir -p "$DATA_DIR"
fi

process_pdfs() {
    cd "$PROJECT_ROOT"
    
    # Check if there are any PDF files in data directory
    pdf_count=$(find "$DATA_DIR" -name "*.pdf" 2>/dev/null | wc -l)
    
    if [ "$pdf_count" -eq 0 ]; then
        echo "No PDFs found in data directory. Using default FAA handbook URL..."
        # Use the direct URL functionality from main.py
        python -m src.main process-pdf -o "$VECTOR_STORE_DIR"
    else
        echo "Found $pdf_count PDF(s) in data directory"
        
        # Process each PDF file
        for pdf_file in "$DATA_DIR"/*.pdf; do
            if [ -f "$pdf_file" ]; then
                echo "Processing: $(basename "$pdf_file")"
                python -m src.main process-pdf "$pdf_file" -o "$VECTOR_STORE_DIR"
            fi
        done
    fi
}

validate_vector_store() {
    if [ -d "$VECTOR_STORE_DIR" ] && [ -f "$VECTOR_STORE_DIR/config.json" ]; then
        echo "Vector store validation: PASSED"
        echo "Vector store contents:"
        ls -la "$VECTOR_STORE_DIR"
    else
        echo "Vector store validation: FAILED"
        echo "Vector store directory or config file missing"
        exit 1
    fi
}

main() {
    echo "Step 1: Processing PDFs and building vector store..."
    process_pdfs
    
    echo "Step 2: Validating vector store..."
    validate_vector_store
    
    echo "=== Vector Store Build Complete ==="
}

main "$@"
