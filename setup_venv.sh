#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up virtual environment for Code Analysis Agents...${NC}"

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${GREEN}Installing required packages...${NC}"
pip install -r requirements.txt

echo -e "${YELLOW}Checking if Ollama is installed...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found! Please install Ollama from https://ollama.ai/${NC}"
    echo -e "${YELLOW}After installing, run: ollama pull codellama${NC}"
else
    echo -e "${GREEN}Ollama is installed. Checking available models...${NC}"
    MODELS=$(ollama list 2>/dev/null || echo "Error listing models")
    
    # Check if we got an error listing models
    if [[ $MODELS == "Error"* ]]; then
        echo -e "${YELLOW}Could not list models. Make sure Ollama service is running.${NC}"
        echo -e "${YELLOW}Run: ollama serve${NC}"
    else
        # Check for recommended models
        RECOMMENDED_MODELS=("codellama" "llama3" "mistral" "mixtral")
        MISSING_MODELS=()
        
        for model in "${RECOMMENDED_MODELS[@]}"; do
            if ! echo "$MODELS" | grep -q "$model"; then
                MISSING_MODELS+=("$model")
            fi
        done
        
        if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
            echo -e "${YELLOW}The following recommended models are not installed:${NC}"
            for model in "${MISSING_MODELS[@]}"; do
                echo -e "  - $model"
            done
            
            echo -e "${YELLOW}Would you like to pull missing models? (y/n)${NC}"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                for model in "${MISSING_MODELS[@]}"; do
                    echo -e "${GREEN}Pulling $model...${NC}"
                    ollama pull "$model"
                done
            fi
        else
            echo -e "${GREEN}All recommended models are installed!${NC}"
        fi
    fi
fi

# Create logs directory
mkdir -p logs

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo -e "${BLUE}Setup complete!${NC}"
    echo -e "${GREEN}To activate the virtual environment, run:${NC}"
    echo -e "source venv/bin/activate"
    echo -e "${GREEN}To use the agent system, run:${NC}"
    echo -e "python -m rag.rag analyze /path/to/code"
    echo -e "python -m rag.rag explain /path/to/code/file.py"
    echo -e "python -m rag.rag doc_query \"how does function X work?\""
    echo -e "python -m rag.rag export \"search query\" --output-format markdown"
    echo -e "python -m rag.rag visualize \"search query\""
else
    echo -e "${RED}Setup failed. Please check the error messages above.${NC}"
fi
