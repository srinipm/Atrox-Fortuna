import os
import json
import chromadb
import typer
from typing import Optional, List, Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
import uuid
import time
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import networkx as nx
# Update the import to use the recommended class
from langchain_ollama import OllamaLLM
# Fix relative import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.code_agents import CodeAgentSystem, logger

app = typer.Typer(help="Code analysis and search tool using AI agents")
console = Console()

class RAG:
    def __init__(self, root_dir, persist_directory="./chroma_db"):
        self.root_dir = root_dir
        self.data = {}
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("code_collection")
        
    def scan_directory(self):
        total_files = sum(len(files) for _, _, files in os.walk(self.root_dir))
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"[green]Scanning {self.root_dir}...", total=total_files)
            
            documents = []
            metadatas = []
            ids = []
            file_count = 0
            
            for dirpath, _, filenames in os.walk(self.root_dir):
                for filename in filenames:
                    file_count += 1
                    progress.update(task, completed=file_count, description=f"[green]Processing {filename}")
                    
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                        
                        # Skip empty files
                        if not content:
                            continue
                        
                        doc_id = str(uuid.uuid4())
                        documents.append(content)
                        metadatas.append({
                            "path": file_path, 
                            "filename": filename,
                            "extension": Path(filename).suffix,
                            "directory": dirpath,
                            "size_bytes": os.path.getsize(file_path),
                            "indexed_at": time.time()
                        })
                        ids.append(doc_id)
                        self.data[doc_id] = {"content": content, "path": file_path}
                        
                    except Exception as e:
                        console.print(f"[bold red]Error reading {file_path}: {e}[/bold red]")
            
            # Add data to collection in batches to avoid memory issues
            if documents:
                batch_size = 100  # Adjust based on available memory
                for i in range(0, len(documents), batch_size):
                    end = min(i + batch_size, len(documents))
                    self.collection.add(
                        documents=documents[i:end],
                        metadatas=metadatas[i:end],
                        ids=ids[i:end]
                    )
                console.print(f"[green]Added {len(documents)} documents to vector database[/green]")
            else:
                console.print("[yellow]No documents found to add[/yellow]")

    def save_to_file(self, output_file):
        try:
            with open(output_file, 'w') as file:
                json.dump(self.data, file)
            console.print(f"[green]Data saved to {output_file}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error writing to {output_file}: {e}[/bold red]")

    def load_from_file(self, input_file):
        try:
            with open(input_file, 'r') as file:
                self.data = json.load(file)
            console.print(f"[green]Data loaded from {input_file}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error reading from {input_file}: {e}[/bold red]")

    def query(self, keyword, n_results=5, min_similarity=0.1):
        """Query the vector database with advanced similarity filtering"""
        results = self.collection.query(
            query_texts=[keyword],
            n_results=n_results * 2  # Get more results than needed to filter by similarity
        )
        
        query_results = {}
        if results and results['ids'] and results['documents']:
            for doc_id, doc, metadata, distance in zip(
                results['ids'][0], 
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            ):
                similarity = 1 - distance
                if similarity >= min_similarity:
                    query_results[metadata['path']] = {
                        'content': doc,
                        'similarity': similarity,
                        'metadata': metadata
                    }
        
        # Sort by similarity and limit to n_results
        sorted_results = dict(sorted(
            query_results.items(), 
            key=lambda item: item[1]['similarity'], 
            reverse=True
        )[:n_results])
        
        return sorted_results

    def export_results(self, results, output_format="json", file_path=None):
        """Export query results to a file in the specified format"""
        if not file_path:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_path = f"query_results_{timestamp}.{output_format}"
        
        try:
            if output_format == "json":
                # Convert results to serializable format
                serializable_results = {}
                for path, data in results.items():
                    serializable_results[path] = {
                        'content': data['content'],
                        'similarity': data['similarity'],
                        'metadata': data.get('metadata', {})
                    }
                
                with open(file_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                
            elif output_format == "csv":
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow(['Path', 'Similarity', 'Extension', 'Size (bytes)', 'Content'])
                    # Write data
                    for path, data in results.items():
                        metadata = data.get('metadata', {})
                        writer.writerow([
                            path, 
                            data['similarity'],
                            metadata.get('extension', ''),
                            metadata.get('size_bytes', ''),
                            data['content'][:200] + '...' if len(data['content']) > 200 else data['content']
                        ])
                        
            elif output_format == "markdown":
                with open(file_path, 'w') as f:
                    f.write("# Query Results\n\n")
                    for path, data in results.items():
                        f.write(f"## {path}\n\n")
                        f.write(f"Similarity: {data['similarity']:.4f}\n\n")
                        f.write("```\n")
                        f.write(data['content'])
                        f.write("\n```\n\n")
            
            console.print(f"[green]Results exported to {file_path}[/green]")
            return file_path
            
        except Exception as e:
            console.print(f"[bold red]Error exporting results: {e}[/bold red]")
            return None

    def visualize_code_relationships(self, results, output_file=None):
        """Generate a graph visualization of code relationships"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            G = nx.DiGraph()
            
            # Build graph from query results
            for path, data in results.items():
                filename = os.path.basename(path)
                G.add_node(filename, path=path, similarity=data['similarity'])
                
                # Look for imports and relationships in the content
                content = data['content']
                for other_path, other_data in results.items():
                    if path != other_path:
                        other_filename = os.path.basename(other_path)
                        # Check if this file imports or references the other file
                        if other_filename in content:
                            G.add_edge(filename, other_filename, weight=1)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G)
            
            # Node color based on similarity
            node_colors = [G.nodes[n].get('similarity', 0.5) for n in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, cmap=plt.cm.Reds, alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20)
            
            plt.title("Code Relationships Graph")
            plt.axis("off")
            
            if output_file:
                plt.savefig(output_file)
                console.print(f"[green]Graph saved to {output_file}[/green]")
            else:
                # Save to temp file
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_file = f"code_graph_{timestamp}.png"
                plt.savefig(output_file)
                console.print(f"[green]Graph saved to {output_file}[/green]")
            
            plt.close()
            return output_file
            
        except Exception as e:
            console.print(f"[bold red]Error generating graph: {e}[/bold red]")
            return None

@app.command()
def index(
    root_dir: str = typer.Argument(..., help="Root directory to scan"),
    output: Optional[str] = typer.Option(None, help="Output file to save the scanned data"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory to persist vector database")
):
    """Index code from the specified directory into vector database."""
    rag = RAG(root_dir, persist_directory=persist_dir)
    rag.scan_directory()
    
    if output:
        rag.save_to_file(output)

@app.command()
def chat(
    persist_dir: str = typer.Option("./chroma_db", help="Directory where vector database is stored")
):
    """Interactive chatbot interface to query the code repository."""
    rag = RAG("", persist_directory=persist_dir)
    
    console.print(Panel.fit(
        "[bold green]Code Repository Chatbot[/bold green]\n"
        "Enter your queries to search the code. Type 'exit' to quit."
    ))
    
    while True:
        query = Prompt.ask("\n[bold cyan]Query[/bold cyan]")
        
        if query.lower() in ('exit', 'quit'):
            break
            
        results = rag.query(query)
        
        if not results:
            console.print("[yellow]No relevant documents found.[/yellow]")
            continue
        
        console.print(f"\n[bold green]Found {len(results)} relevant files:[/bold green]")
        
        for i, (path, data) in enumerate(results.items(), 1):
            similarity = data.get('similarity', 0)
            console.print(f"\n[bold cyan]{i}. {path}[/bold cyan] (Similarity: {similarity:.2f})")
            
            # Display code with syntax highlighting
            language = path.split('.')[-1] if '.' in path else 'text'
            syntax = Syntax(data['content'], language, theme="monokai", line_numbers=True)
            console.print(Panel(syntax, expand=False))

@app.command()
def query(
    keyword: str = typer.Argument(..., help="Keyword to query"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory where vector database is stored"),
    n_results: int = typer.Option(5, help="Number of results to return")
):
    """Query the code repository with a keyword."""
    rag = RAG("", persist_directory=persist_dir)
    results = rag.query(keyword, n_results=n_results)
    
    if not results:
        console.print("[yellow]No relevant documents found.[/yellow]")
        return
    
    console.print(f"\n[bold green]Found {len(results)} relevant files:[/bold green]")
    
    for path, data in results.items():
        similarity = data.get('similarity', 0)
        console.print(f"\n[bold cyan]{path}[/bold cyan] (Similarity: {similarity:.2f})")
        
        # Display code with syntax highlighting
        language = path.split('.')[-1] if '.' in path else 'text'
        syntax = Syntax(data['content'], language, theme="monokai", line_numbers=True)
        console.print(Panel(syntax, expand=False))

@app.command()
def analyze(
    root_dir: str = typer.Argument(..., help="Root directory containing code to analyze"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory to persist vector database"),
    model: str = typer.Option("codellama", help="Ollama model to use for analysis"),
    max_workers: int = typer.Option(4, help="Maximum number of parallel workers")
):
    """Analyze code using AI agents and store documentation in vector database."""
    console.print(f"[bold blue]Starting code analysis on {root_dir} using {model} model with {max_workers} workers...[/bold blue]")
    
    try:
        agent_system = CodeAgentSystem(model_name=model, persist_directory=persist_dir)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False
        ) as progress:
            task = progress.add_task("Analyzing code...", total=None)
            agent_system.analyze_code_path(root_dir, max_workers=max_workers)
            progress.update(task, description="Analysis complete!")
        
        console.print("[bold green]Analysis complete! Documentation has been stored in the vector database.[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {e}[/bold red]")

@app.command()
def explain(
    file_path: str = typer.Argument(..., help="Path to the file you want explained"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory where vector database is stored"),
    model: str = typer.Option("codellama", help="Ollama model to use for explanation")
):
    """Get an AI-generated explanation of a specific file."""
    if not os.path.isfile(file_path):
        console.print(f"[bold red]File not found: {file_path}[/bold red]")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except Exception as e:
        console.print(f"[bold red]Error reading {file_path}: {e}[/bold red]")
        return
    
    console.print(f"[bold blue]Generating explanation for {file_path}...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Analyzing code...[/bold blue]"),
        transient=True,
    ) as progress:
        progress.add_task("Analyzing", total=None)
        # Update to use the new OllamaLLM class
        llm = OllamaLLM(model=model, temperature=0.2)
        explanation = llm.invoke(f"""
        Please explain the following code file. Include:
        1. Overall purpose
        2. Main functions/methods and what they do
        3. Key patterns or algorithms used
        4. Dependencies and how they're used
        
        Code:
        ```
        {content}
        ```
        """)
    
    language = file_path.split('.')[-1] if '.' in file_path else 'text'
    
    console.print("\n[bold green]Code Explanation:[/bold green]")
    console.print(Panel(explanation))
    
    console.print("\n[bold green]Original Code:[/bold green]")
    syntax = Syntax(content, language, theme="monokai", line_numbers=True)
    console.print(Panel(syntax, expand=False))

@app.command()
def doc_query(
    query: str = typer.Argument(..., help="Query for searching code documentation"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory where vector database is stored"),
    n_results: int = typer.Option(5, help="Number of results to return")
):
    """Query the AI-generated code documentation."""
    agent_system = CodeAgentSystem(persist_directory=persist_dir)
    results = agent_system.query_documentation(query, n_results=n_results)
    
    if not results:
        console.print("[yellow]No relevant documentation found.[/yellow]")
        return
    
    console.print(f"\n[bold green]Found documentation in {len(results)} files:[/bold green]")
    
    for path, chunks in results.items():
        console.print(f"\n[bold cyan]{path}[/bold cyan]")
        
        for chunk in chunks:
            console.print(Panel(chunk['content'], title=f"Chunk {chunk['chunk_id']}"))

@app.command()
def export(
    query: str = typer.Argument(..., help="Query to find code for export"),
    output_format: str = typer.Option("json", help="Output format: json, csv, or markdown"),
    output_file: Optional[str] = typer.Option(None, help="Output file path"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory where vector database is stored"),
    n_results: int = typer.Option(5, help="Number of results to return")
):
    """Query the code repository and export results to a file."""
    rag = RAG("", persist_directory=persist_dir)
    results = rag.query(query, n_results=n_results)
    
    if not results:
        console.print("[yellow]No relevant documents found.[/yellow]")
        return
    
    console.print(f"\n[bold green]Found {len(results)} relevant files:[/bold green]")
    
    # Display summary table
    table = Table(title="Search Results")
    table.add_column("File", style="cyan")
    table.add_column("Similarity", style="green")
    table.add_column("Size", style="blue")
    
    for path, data in results.items():
        similarity = data.get('similarity', 0)
        size = data.get('metadata', {}).get('size_bytes', 0)
        table.add_row(os.path.basename(path), f"{similarity:.4f}", f"{size} bytes")
    
    console.print(table)
    
    # Export results
    export_path = rag.export_results(results, output_format=output_format, file_path=output_file)
    
    if export_path:
        console.print(f"[bold green]Results exported to: {export_path}[/bold green]")

@app.command()
def visualize(
    query: str = typer.Argument(..., help="Query to find code for visualization"),
    output_file: Optional[str] = typer.Option(None, help="Output file for graph image"),
    persist_dir: str = typer.Option("./chroma_db", help="Directory where vector database is stored"),
    n_results: int = typer.Option(10, help="Number of results to include in visualization")
):
    """Generate a visualization of code relationships based on query results."""
    rag = RAG("", persist_directory=persist_dir)
    results = rag.query(query, n_results=n_results)
    
    if not results:
        console.print("[yellow]No relevant documents found.[/yellow]")
        return
    
    console.print(f"\n[bold green]Generating visualization for {len(results)} files...[/bold green]")
    
    graph_path = rag.visualize_code_relationships(results, output_file=output_file)
    
    if graph_path:
        console.print(f"[bold green]Visualization saved to: {graph_path}[/bold green]")

if __name__ == "__main__":
    app()
