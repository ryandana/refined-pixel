#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "==============================================="
echo "       AI Image Processor - Starting..."
echo "==============================================="
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

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ main.py not found!${NC}"
    echo "Please ensure all files are in the correct directory."
    exit 1
fi

echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${BLUE}🔄 Checking dependencies...${NC}"
python3 -c "import flask, rembg, cv2, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Dependencies not properly installed!${NC}"
    echo "Please run setup.py again."
    exit 1
fi

echo -e "${GREEN}✅ Dependencies verified${NC}"
echo ""
echo -e "${CYAN}🚀 Starting AI Image Processor...${NC}"
echo -e "${YELLOW}📱 The application will open in your default browser${NC}"
echo -e "${YELLOW}🛑 Press Ctrl+C to stop the server${NC}"
echo ""

# Handle interrupt signal gracefully
trap 'echo -e "\n${YELLOW}👋 Application stopped${NC}"; exit 0' INT

python3 main.py