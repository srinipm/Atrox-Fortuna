# Java Dependency Analyzer

This tool analyzes a Java codebase to build a dependency graph of function and method calls, including recursive call chains and database relationships.

## Prerequisites

- Python 3.6+
- Required dependencies can be installed using:
  ```bash
  pip install -r requirements.txt
  ```

## Setup

### Using Virtual Environment (Recommended)

#### On macOS/Linux:
```bash
# Navigate to the tools directory
cd tools

# Make the setup script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh

# Activate the virtual environment
source venv/bin/activate
```

#### On Windows:
```batch
# Navigate to the tools directory
cd tools

# Run the setup script
setup_venv.bat

# Activate the virtual environment
venv\Scripts\activate
```

### Manual Installation
```bash
pip install -r requirements.txt
```

## Usage

### Analyzing Your Java Code

1. Run the dependency analyzer on your Java codebase:

```bash
python java_dependency_analyzer.py /path/to/your/java/project
```

This will generate two output files:
- `dependency_graph.dot`: GraphViz dot format file
- `dependency_graph.json`: JSON file for web visualization

### Command Line Options

```bash
python java_dependency_analyzer.py --help
```

Options:
- `-o`, `--output`: Specify the output file for the DOT graph (default: dependency_graph.dot)
- `-j`, `--json`: Specify the output file for the JSON data (default: dependency_graph.json)
- `-r`, `--recursive`: Analyze dependencies recursively (default: True)
- `-d`, `--depth`: Maximum recursion depth for dependency analysis (default: 5)
- `--direct-only`: Only analyze direct method calls, not recursive dependencies
- `--no-deep`: Skip deep analysis of database interactions

Database Analysis Options:
- `--no-db`: Exclude database table analysis
- `--db-only`: Focus only on database interactions (methods that access databases)

### Analysis Modes

The tool supports different analysis modes:

1. **Complete Analysis** (default): Shows both method dependencies and database interactions
   ```bash
   python java_dependency_analyzer.py /path/to/project
   ```

2. **Method Dependencies Only**: Excludes database tables and interactions
   ```bash
   python java_dependency_analyzer.py /path/to/project --no-db
   ```

3. **Database Interactions Only**: Shows only methods that interact with databases and their relationships
   ```bash
   python java_dependency_analyzer.py /path/to/project --db-only
   ```

### Visualizing the Dependencies

#### Using GraphViz

If you have GraphViz installed, you can generate an image from the DOT file:

```bash
dot -Tpng dependency_graph.dot -o dependency_graph.png
```

#### Using the Web Visualizer

1. Open `visualize_dependencies.html` in a web browser
2. Click "Choose File" and select the generated JSON file
3. Explore the dependency graph interactively:
   - Zoom in/out with the mouse wheel
   - Drag nodes to reposition them
   - Hover over nodes to see details
   - Filter by package name
   - Adjust visualization depth with the slider

## Features

- **Identifies method dependencies** across your Java codebase
- **Tracks recursive call chains** to find indirect dependencies
- **Detects database tables** referenced in SQL queries
- **Identifies database operations** (SELECT, INSERT, UPDATE, DELETE)
- **Detects implicit database interactions** through various means:
  - SQL string constants
  - String concatenation that builds SQL
  - Database-related annotations
  - ORM framework methods
  - Common database access patterns
- **Follows indirect database access** through call chains
- **Recognizes database-related classes** by name patterns and fields
- **Generates visualizations** in multiple formats
- **Interactive web visualization** with filtering and exploration tools
- **Supports large codebases** with efficient parsing

## Advanced Database Detection

The tool uses multiple strategies to detect database interactions:

1. **Direct SQL String Detection**:
   - Identifies SQL queries in string literals
   - Parses SQL to extract table names and operation types

2. **String Constant Analysis**:
   - Detects class-level SQL constants
   - Tracks their usage throughout the codebase

3. **String Concatenation Analysis**:
   - Identifies SQL queries built through string concatenation
   - Analyzes StringBuilder/StringBuffer usage for SQL building

4. **Database-Related Patterns**:
   - Recognizes classes with database-related names (Repository, DAO, etc.)
   - Identifies methods with database access patterns

5. **Annotation-Based Detection**:
   - Recognizes JPA, Spring, and other database-related annotations
   - Uses annotations to infer database interactions

6. **Implicit Database Access**:
   - Follows call chains to identify methods indirectly accessing databases
   - Propagates database table access information through the call graph

7. **Framework Method Recognition**:
   - Detects common JDBC, JPA, Hibernate, MyBatis method calls
   - Identifies Spring Data and other ORM framework patterns

## Limitations

- Method resolution relies on static analysis and may not be 100% accurate for dynamic calls
- Resolution of method calls through interfaces and inheritance is limited
- External library dependencies are not analyzed
- SQL query extraction may miss dynamically constructed queries
- Database table detection may not work for complex or nonstandard SQL
- Detection of implicit database access may produce false positives

## Example

Running the analyzer on a sample project:

```bash
python java_dependency_analyzer.py /Users/srinivasanpichumani/github/Atrox-Fortuna/sybase/samples
```

Then view the results:

```bash
dot -Tpng dependency_graph.dot -o dependency_graph.png
open dependency_graph.png  # On macOS
```

Or use the web visualizer by opening `visualize_dependencies.html` in your browser.
