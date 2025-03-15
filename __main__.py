"""
Main entry point when running the package with python -m
"""
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag import app

if __name__ == "__main__":
    app()
