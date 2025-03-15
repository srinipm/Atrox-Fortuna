import os
import sys
import logging
from crewai import Agent, Task, Crew, Process
# Update the import to use the recommended class
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import chromadb
from pathlib import Path
import tempfile
import concurrent.futures
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("code_analysis.log")
    ]
)
logger = logging.getLogger('code_agents')

class CodeAgentSystem:
    def __init__(self, model_name="codellama", persist_directory="./chroma_db", temperature=0.2):
        """Initialize the code agent system with Ollama model and vector store"""
        self.model_name = model_name
        try:
            # Update to use the new OllamaLLM class instead of the deprecated Ollama
            self.llm = OllamaLLM(model=model_name, temperature=temperature)
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.code_collection = self.client.get_or_create_collection("code_collection")
            self.docs_collection = self.client.get_or_create_collection("documentation_collection")
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            logger.info(f"Initialized CodeAgentSystem with model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize CodeAgentSystem: {str(e)}")
            raise
            
        # Supported file extensions
        self.supported_extensions = (
            # Python
            '.py', '.pyx', '.pyi', '.ipynb',
            # Web
            '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.vue', '.svelte',
            # Backend
            '.go', '.java', '.kt', '.cs', '.php', '.rb', '.rs', '.scala',
            # Systems
            '.c', '.cpp', '.cc', '.h', '.hpp', '.swift',
            # Data
            '.sql', '.graphql',
            # Config
            '.json', '.yaml', '.yml', '.toml'
        )

    def create_agents(self):
        """Create the agents for code analysis"""
        code_analyzer = Agent(
            role="Code Analyzer",
            goal="Analyze code files to understand their structure and functionality",
            backstory="You're an expert code analyzer with years of experience understanding complex codebases.",
            verbose=True,
            llm=self.llm,
            tools=[]
        )
        
        documenter = Agent(
            role="Technical Documentation Writer",
            goal="Create clear and comprehensive documentation for code functionality",
            backstory="You're a technical writer who specializes in creating documentation for software projects.",
            verbose=True,
            llm=self.llm,
            tools=[]
        )
        
        indexer = Agent(
            role="Code Indexer",
            goal="Organize and structure code information for efficient retrieval",
            backstory="You're a data organization specialist who creates efficient indexing systems.",
            verbose=True,
            llm=self.llm,
            tools=[]
        )
        
        return code_analyzer, documenter, indexer
    
    def analyze_code_path(self, code_path, max_workers=4):
        """Analyze all code files in a given directory path with parallel processing"""
        analyzer, documenter, indexer = self.create_agents()
        
        if os.path.isdir(code_path):
            file_paths = []
            for root, _, files in os.walk(code_path):
                for file in files:
                    if file.endswith(self.supported_extensions):
                        file_paths.append(os.path.join(root, file))
            logger.info(f"Found {len(file_paths)} files to analyze in {code_path}")
        else:
            file_paths = [code_path]
        
        # Process files in parallel with a maximum number of workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            futures = {executor.submit(self._process_file, file_path, analyzer, documenter, indexer): file_path 
                      for file_path in file_paths}
            
            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Analyzing files"):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
    
    def _process_file(self, file_path, analyzer, documenter, indexer):
        """Process a single file with the agent crew"""
        if not os.path.isfile(file_path):
            return
                
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            
            # Skip empty files or files too large
            if not code_content or len(code_content) > 100000:
                logger.warning(f"Skipping {file_path}: empty or too large ({len(code_content) if code_content else 0} characters)")
                return
            
            file_extension = os.path.splitext(file_path)[1].lower()
            file_type = self._determine_file_type(file_extension)
            
            # Create analysis task with file type context and expected_output
            analysis_task = Task(
                description=f"""Analyze the {file_type} code in file {os.path.basename(file_path)} and identify its key components, functionality, and structure.
                
                Focus on:
                1. Purpose of this file
                2. Key functions/classes/components
                3. Dependencies and imports
                4. Control flow and logic
                5. Notable patterns or algorithms
                
                File: {os.path.basename(file_path)}
                Path: {file_path}
                Type: {file_type}
                
                Code:
                ```{file_extension[1:] if file_extension else ''}
                {code_content}
                ```
                """,
                expected_output="A detailed analysis of the code file including its structure, components, and functionality.",
                agent=analyzer
            )
            
            # Create documentation task with expected_output
            documentation_task = Task(
                description="""Create comprehensive documentation for the analyzed code. 
                
                Include:
                1. Overview and purpose
                2. Details of each function/method/class
                3. Dependencies and relationships
                4. Usage examples
                5. Potential issues or optimizations
                
                Format your documentation in markdown with appropriate sections and code examples.
                """,
                expected_output="Comprehensive markdown documentation of the code with sections for overview, functions/methods, dependencies, and examples.",
                agent=documenter,
                depends_on=[analysis_task]
            )
            
            # Create indexing task with expected_output
            indexing_task = Task(
                description="""Create searchable index entries for the code and documentation.
                
                Focus on:
                1. Key concepts and terminology
                2. Function signatures and their purpose
                3. Interface definitions
                4. Design patterns implemented
                5. Tags for efficient searching
                
                Structure your index to maximize searchability and context.
                """,
                expected_output="A structured set of index entries with keywords, concepts, and code elements for efficient searching.",
                agent=indexer,
                depends_on=[documentation_task]
            )
            
            # Create and run the crew for this file
            file_crew = Crew(
                agents=[analyzer, documenter, indexer],
                tasks=[analysis_task, documentation_task, indexing_task],
                process=Process.sequential,
                verbose=False  # Reduce console clutter during batch processing
            )
            
            logger.info(f"Starting analysis of {file_path}")
            result = file_crew.kickoff()
            logger.info(f"Completed analysis of {file_path}")
            
            # Store the documentation in the vector database
            self._store_documentation(file_path, result)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def _determine_file_type(self, extension):
        """Determine the general type of file based on its extension"""
        extension = extension.lower()
        
        file_type_map = {
            '.py': 'Python',
            '.pyx': 'Cython',
            '.ipynb': 'Jupyter Notebook',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React JavaScript',
            '.tsx': 'React TypeScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.vue': 'Vue.js',
            '.svelte': 'Svelte',
            '.go': 'Go',
            '.java': 'Java',
            '.kt': 'Kotlin',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.rs': 'Rust',
            '.c': 'C',
            '.cpp': 'C++',
            '.cc': 'C++',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.swift': 'Swift',
            '.sql': 'SQL',
            '.graphql': 'GraphQL',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.scala': 'Scala'
        }
        
        return file_type_map.get(extension, 'Unknown')
    
    def _store_documentation(self, file_path, documentation):
        """Store documentation in the vector database"""
        chunks = self.text_splitter.split_text(documentation)
        
        # Store each chunk in the vector database
        for i, chunk in enumerate(chunks):
            self.docs_collection.add(
                documents=[chunk],
                metadatas=[{
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "chunk_id": i
                }],
                ids=[f"{Path(file_path).stem}_doc_{i}"]
            )
    
    def query_documentation(self, query, n_results=5):
        """Query the documentation database"""
        results = self.docs_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        query_results = {}
        if results and results['ids'] and results['documents']:
            for doc_id, doc, metadata in zip(
                results['ids'][0], 
                results['documents'][0], 
                results['metadatas'][0]
            ):
                file_path = metadata['path']
                if file_path not in query_results:
                    query_results[file_path] = []
                
                query_results[file_path].append({
                    'content': doc,
                    'chunk_id': metadata.get('chunk_id')
                })
        
        return query_results
    
    def batch_analyze(self, code_paths, max_workers=None):
        """Analyze multiple code paths in batch"""
        if max_workers is None:
            # Default to number of CPUs
            max_workers = os.cpu_count() or 4
        
        start_time = datetime.now()
        logger.info(f"Starting batch analysis of {len(code_paths)} paths at {start_time}")
        
        for path in code_paths:
            self.analyze_code_path(path, max_workers=max_workers)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Completed batch analysis in {duration}")
        
        return {
            "paths_analyzed": len(code_paths),
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration.total_seconds()
        }
