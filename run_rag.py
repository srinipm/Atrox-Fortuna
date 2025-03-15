#!/usr/bin/env python3
"""
Simple runner script for the RAG application.
This avoids package import issues by launching the application directly.
"""
import os
import sys
import traceback

# Ensure the parent directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        # Import and run the app
        from rag.rag import app
        app()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you have installed all required packages:")
        print("   pip install -r requirements.txt")
        print("2. Check if the correct version of crewai is installed:")
        print("   pip show crewai")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
