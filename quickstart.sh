#!/bin/bash
# Quick start script for AI Contraception Counseling System

echo "========================================="
echo "AI Contraception Counseling System Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
echo "Virtual environment created!"
echo ""

# Activate virtual environment
echo "To activate the virtual environment, run:"
echo "  Windows: venv\\Scripts\\activate"
echo "  Mac/Linux: source venv/bin/activate"
echo ""

# Install dependencies
read -p "Would you like to install dependencies now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing dependencies..."
    source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Dependencies installed!"
fi
echo ""

# Set up environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo ".env file created! Please edit it and add your API keys."
else
    echo ".env file already exists."
fi
echo ""

# Create necessary directories
echo "Ensuring all directories exist..."
mkdir -p data/{who,bcs,synthetic,processed,memory}
mkdir -p results/{tables,plots,logs}
echo "Directories created!"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment"
echo "2. Edit .env file and add your API keys"
echo "3. Review and customize configs/config.yaml"
echo "4. Place WHO and BCS+ PDFs in data/who/ and data/bcs/"
echo "5. Run data preprocessing: python src/rag/preprocess_documents.py"
echo "6. Start the API: uvicorn src.api.main:app --reload"
echo ""
echo "For more information, see README.md"
echo ""
