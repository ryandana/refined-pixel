#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "================================================"
echo "    AI Image Processor - Starting (CUDA)..."
echo "================================================"
echo -e "${NC}"

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run setup.py first to install dependencies."
    echo ""
    echo "python3 setup.py"
    echo ""
    exit 1
fi

# Check CUDA availability
echo -e "${BLUE}🔄 Checking CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi &>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        echo -e "${PURPLE}🔥 CUDA acceleration enabled${NC}"
    else
        echo -e "${YELLOW}⚠️  NVIDIA GPU detected but drivers may not be working${NC}"
        echo -e "${YELLOW}Falling back to CPU mode...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU not detected or nvidia-smi not found${NC}"
    echo -e "${YELLOW}Falling back to CPU mode...${NC}"
fi

echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${BLUE}🔄 Checking PyTorch CUDA support...${NC}"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ PyTorch not properly installed!${NC}"
    echo "Please run setup.py again with CUDA support."
    exit 1
fi

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For better performance on some systems
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo ""
echo -e "${CYAN}🚀 Starting AI Image Processor with CUDA...${NC}"
echo -e "${YELLOW}📱 The application will open in your default browser${NC}"
echo -e "${YELLOW}🛑 Press Ctrl+C to stop the server${NC}"
echo ""

# Handle interrupt signal gracefully
trap 'echo -e "\n${YELLOW}👋 Application stopped${NC}"; exit 0' INT

python3 main.py