# Code Analysis Agent System

A CrewAI-based agentic system for analyzing, documenting, and searching code repositories using vector databases and local LLMs. This tool helps developers understand complex codebases by leveraging AI to generate comprehensive documentation, enable semantic search, and visualize code relationships.

## Features

- **Intelligent Code Analysis**: Analyze code using AI agents powered by Ollama's local models
- **Automated Documentation**: Generate comprehensive documentation with context and examples
- **Vector Search**: Store and semantically search code and documentation in ChromaDB
- **Interactive CLI**: Query and interact with your codebase through a rich terminal interface
- **Code Visualization**: Generate visual graphs of code relationships and dependencies
- **Export Capabilities**: Export search results in JSON, CSV, or Markdown formats
- **Multi-threaded Processing**: Parallel processing for faster analysis of large codebases

## Prerequisites

- **Python 3.9+** (recommended for best compatibility with all dependencies)
- **[Ollama](https://ollama.ai/)** - Local LLM runner (version 0.1.20+)
- At least 8GB RAM (16GB+ recommended for larger codebases)
- 2GB+ of free disk space (for model storage and vector database)

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Atrox-Fortuna.git
cd Atrox-Fortuna

# Set up the virtual environment and install dependencies
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

### Manual Installation

If you prefer to set up manually:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Ensure Ollama is installed and download required models
ollama pull codellama
```

## Usage

### Run with the Convenience Script (Recommended)

The simplest way to use the tool:

```bash
python run_rag.py [command] [arguments]
```

### Run as a Python Module

For standard Python module usage:

```bash
python -m rag [command] [arguments]
```

### Environmental Variables

You can configure the tool using environment variables:

```bash
# Set default model
export RAG_MODEL=codellama

# Set database location
export RAG_DB_DIR=/path/to/database
```

## Commands

### Index Your Codebase

Store code in the vector database for searching:

```bash
python run_rag.py index /path/to/your/code --persist-dir ./my_project_db
```

### Analyze Code with AI Agents

Perform deep analysis of code structure and functionality:

```bash
python run_rag.py analyze /path/to/your/code --model codellama --max-workers 4
```

### Explain a Specific File

Get an AI-generated explanation for a single file:

```bash
python run_rag.py explain /path/to/code/file.py --model codellama
```

### Interactive Chat Mode

Start an interactive session to query your codebase:

```bash
python run_rag.py chat --persist-dir ./my_project_db
```

### Search Specific Code Patterns

Query the codebase for specific patterns or concepts:

```bash
python run_rag.py query "authentication implementation" --n-results 10
```

### Query Generated Documentation

Search through AI-generated code documentation:

```bash
python run_rag.py doc_query "How does the user authentication system work?"
```

### Export Search Results

Export query results to different formats:

```bash
python run_rag.py export "database connection" --output-format markdown --output-file db_connections.md
```

### Visualize Code Relationships

Generate a visual graph of related code components:

```bash
python run_rag.py visualize "payment processing" --output-file payment_system.png
```

## Advanced Options

### Common Parameters

- `--persist-dir`: Custom directory to store the vector database (default: ./chroma_db)
- `--model`: Choose Ollama model (default: codellama)
- `--n-results`: Number of results to return in queries (default: 5)
- `--output-format`: Format for exported results (json, csv, markdown)
- `--max-workers`: Number of parallel workers for analysis (default: CPU count)

### Supported Models

The system works with any model available in Ollama, but these are recommended:

- **codellama**: Best for code analysis and understanding (default)
- **llama3**: Good general performance with balanced capabilities
- **mistral**: Smaller and faster model, good for resource-constrained environments
- **mixtral**: High quality for larger systems and complex analysis

### Troubleshooting

#### Import Issues

If you experience import errors, try:

```bash
# Method 1: Run using the provided script
python run_rag.py [command]

# Method 2: Set PYTHONPATH environment variable
export PYTHONPATH=$PYTHONPATH:/path/to/Atrox-Fortuna
python -m rag.rag [command]
```

#### Ollama Connection Errors

Ensure Ollama service is running:

```bash
# Start Ollama service
ollama serve
```

## Project Structure
