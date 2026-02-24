#!/bin/bash
# build.sh - Quick build script for runtime_converter

set -e

echo "======================================="
echo "Building Runtime Converter"
echo "======================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info
rm -f runtime_converter*.so

# Build extension in-place
echo "Building extension..."
python setup.py build_ext --inplace

# Test import
echo "Testing import..."
python -c "import runtime_converter; print('✓ Build successful!')" || {
    echo "✗ Failed to import runtime_converter"
    exit 1
}

echo ""
echo "======================================="
echo "Build complete!"
echo "======================================="
echo ""
echo "Run example:"
echo "  python example_usage.py"
echo ""