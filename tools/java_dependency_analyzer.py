#!/usr/bin/env python3
"""
Java Dependency Analyzer

This script traverses a Java codebase to build a dependency graph of function and method calls.
The graph is output in DOT format for visualization with tools like Graphviz.
"""

import os
import re
import argparse
from pathlib import Path
import javalang  # You might need to run: pip install javalang
from collections import defaultdict, deque
import sqlparse  # You might need to run: pip install sqlparse


class JavaDependencyAnalyzer:
    def __init__(self):
        self.dependency_graph = {}
        self.recursive_dependencies = defaultdict(set)  # For storing full call tree
        self.class_methods = {}
        self.package_classes = {}
        self.all_methods = set()
        self.call_depths = {}  # Track call depth for visualization
        
        # Database-related tracking
        self.db_tables = set()  # All database tables found
        self.method_table_access = defaultdict(set)  # Method to tables mapping
        self.table_access_types = defaultdict(set)  # (method, table) to access types mapping
        self.sql_queries = defaultdict(list)  # Method to SQL queries mapping
        self.string_constants = {}  # Class to string constants mapping
        self.db_related_methods = set()  # Methods that seem database-related
        
        # Regex patterns for SQL detection
        self.sql_string_pattern = re.compile(
            r'(?:"|\')(?:\s*)?'  # Quote start with optional whitespace
            r'(?i)'  # Case insensitive
            r'(?:\s*)?(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE|MERGE\s+INTO|TRUNCATE\s+TABLE|DROP\s+TABLE|EXEC(?:UTE)?\s+|CALL\s+|WITH|REPLACE|UPSERT)'  # SQL keywords
            r'(?:.*?)(?:"|\')',  # Rest of the query
            re.DOTALL
        )
        
        # Common JDBC method patterns that might contain SQL
        self.jdbc_methods = {
            'executeQuery', 'executeUpdate', 'execute', 'prepareStatement', 'createStatement'
        }

        # Expanded SQL keywords for better detection
        self.sql_keywords = {
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'TRUNCATE', 'MERGE',
            'JOIN', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT',
            'EXEC', 'EXECUTE', 'CALL', 'WITH', 'REPLACE', 'UPSERT'
        }
        
        # Expanded JDBC and common ORM method patterns
        self.db_access_methods = {
            # JDBC
            'executeQuery', 'executeUpdate', 'execute', 'prepareStatement', 'createStatement',
            'prepareCall', 'getConnection', 'setAutoCommit', 'commit', 'rollback',
            'getResultSet', 'getMetaData', 'getGeneratedKeys',
            
            # JPA / Hibernate
            'persist', 'merge', 'find', 'remove', 'createQuery', 'createNamedQuery',
            'createNativeQuery', 'createStoredProcedureQuery', 'detach', 'refresh',
            'getTransaction', 'flush',
            
            # MyBatis
            'selectOne', 'selectList', 'insert', 'update', 'delete',
            
            # Spring JDBC
            'query', 'queryForObject', 'queryForList', 'update', 'execute',
            
            # Other common patterns
            'save', 'saveAll', 'findById', 'findAll', 'deleteById', 'deleteAll'
        }
        
        # Database-related class patterns
        self.db_class_patterns = [
            r'.*Repository', r'.*Dao', r'.*JdbcTemplate', r'.*EntityManager',
            r'.*SessionFactory', r'.*Connection', r'.*DataSource',
            r'.*SqlSession', r'.*JdbcOperations'
        ]
        self.db_related_classes = set()

    def scan_directory(self, directory_path):
        """Scan a directory recursively for Java files"""
        print(f"Scanning directory: {directory_path}")
        java_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        return java_files

    def parse_java_file(self, file_path):
        """Parse a Java file and extract class and method information with enhanced DB detection"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract SQL strings directly from the file content
            self.extract_sql_from_content(content, file_path)
            
            # Parse Java source code
            tree = javalang.parse.parse(content)
            file_package = tree.package.name if tree.package else "default"
            
            # Process class-level string constants (potential SQL)
            self.extract_class_constants(tree, file_package)
            
            # Process each class/interface declaration in the file
            for path, node in tree.filter(javalang.tree.TypeDeclaration):
                class_name = node.name
                qualified_class_name = f"{file_package}.{class_name}"
                self.package_classes.setdefault(file_package, set()).add(class_name)
                
                # Check if class is DB-related based on name patterns
                for pattern in self.db_class_patterns:
                    if re.match(pattern, class_name):
                        self.db_related_classes.add(qualified_class_name)
                        break
                
                # Extract annotations that might indicate DB interaction
                has_db_annotations = self.check_for_db_annotations(node)
                if has_db_annotations:
                    self.db_related_classes.add(qualified_class_name)
                
                # Process each method in the class
                for method_node in node.methods:
                    method_name = method_node.name
                    qualified_method_name = f"{qualified_class_name}.{method_name}"
                    self.all_methods.add(qualified_method_name)
                    self.class_methods.setdefault(qualified_class_name, set()).add(method_name)
                    
                    # Check if method is in DB-related class or has DB-related name
                    if qualified_class_name in self.db_related_classes or method_name in self.db_access_methods:
                        self.db_related_methods.add(qualified_method_name)
                    
                    # Find method calls within this method
                    self.dependency_graph[qualified_method_name] = set()
                    self.extract_method_calls(method_node, qualified_class_name, file_package, qualified_method_name)
                    
                    # Extract SQL from method body 
                    if method_node.body:
                        method_body = " ".join([str(stmt) for stmt in method_node.body])
                        self.extract_sql_from_method(method_body, qualified_method_name)
                        
                        # Check for string concatenations that might be SQL
                        self.analyze_potential_sql_concatenation(method_body, qualified_method_name)
            
            # Further analysis of class-wide database usage patterns
            self.analyze_class_db_patterns(tree, file_package)
            
            return True
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return False

    def extract_sql_from_content(self, content, file_path):
        """Extract SQL queries from file content using regex"""
        # Extract SQL patterns from the content
        sql_matches = self.sql_string_pattern.findall(content)
        if sql_matches:
            print(f"Found {len(sql_matches)} potential SQL patterns in {file_path}")
            # We'll associate these with methods during detailed parsing

    def extract_sql_from_method(self, method_body, qualified_method_name):
        """Extract SQL queries from method body"""
        # Look for SQL strings
        sql_matches = self.sql_string_pattern.findall(method_body)
        
        for sql in sql_matches:
            # Clean and parse the SQL
            sql_clean = sql.strip('\'"')
            self.sql_queries[qualified_method_name].append(sql_clean)
            
            # Extract table names from SQL
            tables = self.extract_tables_from_sql(sql_clean)
            
            for table in tables:
                self.db_tables.add(table)
                self.method_table_access[qualified_method_name].add(table)
                
                # Determine access type
                access_type = self.determine_sql_access_type(sql_clean)
                self.table_access_types[(qualified_method_name, table)].add(access_type)

    def determine_sql_access_type(self, sql):
        """Determine the type of SQL operation (SELECT, INSERT, UPDATE, DELETE)"""
        sql_upper = sql.upper()
        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        elif 'CREATE TABLE' in sql_upper:
            return 'CREATE'
        elif 'ALTER TABLE' in sql_upper:
            return 'ALTER'
        else:
            return 'OTHER'

    def extract_tables_from_sql(self, sql):
        """Extract table names from SQL query using sqlparse"""
        tables = set()
        try:
            # Parse the SQL statement
            parsed = sqlparse.parse(sql)
            if not parsed:
                return tables
                
            stmt = parsed[0]
            
            # Different extraction based on statement type
            if stmt.get_type() == 'SELECT':
                # Look for FROM clause and tables
                from_seen = False
                for token in stmt.tokens:
                    if from_seen and token.ttype is None:
                        # This might be table reference(s)
                        tables.update(self._extract_table_identifiers(token))
                    if token.match(sqlparse.tokens.Keyword, 'FROM'):
                        from_seen = True
            
            elif stmt.get_type() == 'INSERT':
                # Get table name after "INSERT INTO"
                into_seen = False
                for token in stmt.tokens:
                    if into_seen and token.ttype is None:
                        tables.update(self._extract_table_identifiers(token))
                        break
                    if token.match(sqlparse.tokens.Keyword, 'INTO'):
                        into_seen = True
            
            elif stmt.get_type() == 'UPDATE':
                # First table identifier after UPDATE is the table name
                update_seen = False
                for token in stmt.tokens:
                    if update_seen and token.ttype is None:
                        tables.update(self._extract_table_identifiers(token))
                        break
                    if token.match(sqlparse.tokens.Keyword, 'UPDATE'):
                        update_seen = True
            
            elif stmt.get_type() == 'DELETE':
                # Look for table after FROM
                from_seen = False
                for token in stmt.tokens:
                    if from_seen and token.ttype is None:
                        tables.update(self._extract_table_identifiers(token))
                        break
                    if token.match(sqlparse.tokens.Keyword, 'FROM'):
                        from_seen = True
            
            # Fallback using regex for tables we couldn't identify
            if not tables:
                # Common patterns for table names
                table_patterns = [
                    r'FROM\s+([a-zA-Z0-9_\.]+)',
                    r'JOIN\s+([a-zA-Z0-9_\.]+)',
                    r'INTO\s+([a-zA-Z0-9_\.]+)',
                    r'UPDATE\s+([a-zA-Z0-9_\.]+)',
                    r'TABLE\s+([a-zA-Z0-9_\.]+)'
                ]
                
                for pattern in table_patterns:
                    matches = re.findall(pattern, sql, re.IGNORECASE)
                    tables.update(matches)
                    
            # Clean up table names (remove any schema prefixes, etc.)
            cleaned_tables = set()
            for table in tables:
                # Remove quotes, brackets, and split on dots to get the table name
                table = table.strip('`"[]\'')
                if '.' in table:
                    # If we have schema.table, just get the table part
                    table = table.split('.')[-1]
                cleaned_tables.add(table)
                
            return cleaned_tables
            
        except Exception as e:
            print(f"Error extracting tables from SQL: {str(e)} - SQL: {sql[:100]}...")
            return tables

    def _extract_table_identifiers(self, token):
        """Helper method to extract table identifiers from a SQL token"""
        tables = set()
        if hasattr(token, 'tokens'):
            # This is a group of tokens
            for sub_token in token.tokens:
                if sub_token.ttype is None and not sub_token.is_whitespace:
                    # Extract the identifier as a table name
                    table_name = sub_token.value.strip('`"\' ')
                    if table_name and not self._is_sql_keyword(table_name):
                        tables.add(table_name)
                    break  # Usually the first identifier is the table
        elif token.ttype is None:
            table_name = token.value.strip('`"\' ')
            if table_name and not self._is_sql_keyword(table_name):
                tables.add(table_name)
        return tables

    def _is_sql_keyword(self, word):
        """Check if word is an SQL keyword to avoid treating it as a table name"""
        keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'AS', 'ON', 'AND', 'OR', 'NOT'
        }
        return word.upper() in keywords

    def extract_method_calls(self, method_node, current_class, current_package, caller_method):
        """Extract method calls from a method body with enhanced DB detection"""
        for path, node in method_node.filter(javalang.tree.MethodInvocation):
            called_method = node.member
            
            # Check if this is a known DB access method
            if called_method in self.db_access_methods:
                # Try to extract SQL from arguments
                self.extract_sql_from_jdbc_call(node, caller_method)
                # Mark this method as DB-related
                self.db_related_methods.add(caller_method)
            
            # Try to resolve if this is a call to a string constant that might contain SQL
            if called_method == 'prepareStatement' or called_method == 'createQuery' or called_method == 'execute':
                self.check_for_sql_constant_usage(node, current_class, caller_method)
            
            # Try to resolve the target class
            target = None
            if node.qualifier:
                # This is a qualified method call (e.g., obj.method())
                # Need more complex resolution logic here depending on imports, etc.
                if "." in node.qualifier:
                    # Fully qualified name
                    target = node.qualifier
                else:
                    # Could be local variable, class name, etc.
                    # For simplicity, we'll check if it matches a known class
                    for pkg, classes in self.package_classes.items():
                        if node.qualifier in classes:
                            target = f"{pkg}.{node.qualifier}"
                            break
            
            if target:
                qualified_called_method = f"{target}.{called_method}"
            else:
                # Assume it's a method in the current class
                qualified_called_method = f"{current_class}.{called_method}"
            
            self.dependency_graph[caller_method].add(qualified_called_method)
            
            # If we're calling a DB-related method, mark the caller as DB-related
            if qualified_called_method in self.db_related_methods:
                self.db_related_methods.add(caller_method)

    def extract_sql_from_jdbc_call(self, node, caller_method):
        """Extract SQL from JDBC method call arguments"""
        try:
            if not hasattr(node, 'arguments') or not node.arguments:
                return
            
            # Check for string literals in arguments
            for arg in node.arguments:
                if isinstance(arg, javalang.tree.Literal) and hasattr(arg, 'value') and arg.value and \
                   (arg.value.startswith('"') or arg.value.startswith("'")):
                    # This is a string literal that might be SQL
                    sql = arg.value.strip('\'"')
                    if self._looks_like_sql(sql):
                        self.sql_queries[caller_method].append(sql)
                        tables = self.extract_tables_from_sql(sql)
                        for table in tables:
                            self.db_tables.add(table)
                            self.method_table_access[caller_method].add(table)
                            access_type = self.determine_sql_access_type(sql)
                            self.table_access_types[(caller_method, table)].add(access_type)
        except Exception as e:
            print(f"Error extracting SQL from JDBC call: {str(e)}")

    def check_for_sql_constant_usage(self, node, current_class, caller_method):
        """Check if a method is using a constant that contains SQL"""
        if not hasattr(node, 'arguments') or not node.arguments:
            return
            
        for arg in node.arguments:
            # Check for field access which might be a SQL constant
            if isinstance(arg, javalang.tree.MemberReference):
                # Try to resolve the constant
                if current_class in self.string_constants and arg.member in self.string_constants[current_class]:
                    constant_value = self.string_constants[current_class][arg.member]
                    if self._looks_like_sql(constant_value):
                        # Process as SQL
                        tables = self.extract_tables_from_sql(constant_value)
                        for table in tables:
                            self.db_tables.add(table)
                            self.method_table_access[caller_method].add(table)
                            access_type = self.determine_sql_access_type(constant_value)
                            self.table_access_types[(caller_method, table)].add(access_type)

    def _looks_like_sql(self, text):
        """Enhanced check if a string looks like SQL"""
        if not text:
            return False
            
        text_upper = text.upper()
        
        # Quick check for common SQL keywords
        if any(keyword in text_upper for keyword in self.sql_keywords):
            return True
            
        # More detailed pattern matching
        sql_patterns = [
            r'SELECT\s+.+?\s+FROM',
            r'INSERT\s+INTO\s+.+?\s+VALUES',
            r'UPDATE\s+.+?\s+SET',
            r'DELETE\s+FROM',
            r'CREATE\s+TABLE',
            r'ALTER\s+TABLE',
            r'DROP\s+TABLE',
            r'TRUNCATE\s+TABLE',
            r'MERGE\s+INTO',
            r'JOIN\s+\w+\s+ON',
            r'WHERE\s+\w+\s*[=<>]'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text_upper):
                return True
                
        return False

    def analyze_potential_sql_concatenation(self, method_body, qualified_method_name):
        """Look for potential SQL being built via string concatenation"""
        # Common SQL concatenation patterns
        sql_concat_patterns = [
            r'["\']SELECT["\'].*?\+',
            r'["\']INSERT["\'].*?\+',
            r'["\']UPDATE["\'].*?\+',
            r'["\']DELETE["\'].*?\+',
            r'["\']FROM["\'].*?\+',
            r'["\']WHERE["\'].*?\+',
            r'\+.*?["\']SELECT["\']',
            r'\+.*?["\']FROM["\']',
            r'\+.*?["\']WHERE["\']',
            r'StringBuilder.*?append.*?["\']SELECT["\']',
            r'StringBuilder.*?append.*?["\']FROM["\']',
            r'StringBuffer.*?append.*?["\']SELECT["\']'
        ]
        
        for pattern in sql_concat_patterns:
            if re.search(pattern, method_body, re.IGNORECASE | re.DOTALL):
                print(f"Potential SQL concatenation found in {qualified_method_name}")
                
                # Mark this method as likely DB-related
                self.db_related_methods.add(qualified_method_name)
                
                # Try to estimate the table name from context
                # This is imperfect but might catch some cases
                table_patterns = [
                    r'FROM\s+([a-zA-Z0-9_\.]+)',
                    r'JOIN\s+([a-zA-Z0-9_\.]+)',
                    r'INTO\s+([a-zA-Z0-9_\.]+)',
                    r'UPDATE\s+([a-zA-Z0-9_\.]+)',
                    r'TABLE\s+([a-zA-Z0-9_\.]+)'
                ]
                
                for tp in table_patterns:
                    matches = re.findall(tp, method_body, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            potential_table = match.strip('"`\'[]')
                            if '.' in potential_table:
                                potential_table = potential_table.split('.')[-1]
                            
                            # If it looks like a valid table name
                            if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', potential_table):
                                print(f"  - Potential table reference: {potential_table}")
                                self.db_tables.add(potential_table)
                                self.method_table_access[qualified_method_name].add(potential_table)
                                
                                # Try to guess the operation type
                                if 'SELECT' in method_body.upper():
                                    self.table_access_types[(qualified_method_name, potential_table)].add('SELECT')
                                if 'INSERT' in method_body.upper():
                                    self.table_access_types[(qualified_method_name, potential_table)].add('INSERT')
                                if 'UPDATE' in method_body.upper():
                                    self.table_access_types[(qualified_method_name, potential_table)].add('UPDATE')
                                if 'DELETE' in method_body.upper():
                                    self.table_access_types[(qualified_method_name, potential_table)].add('DELETE')
                
                break

    def analyze_class_db_patterns(self, tree, file_package):
        """Analyze class-wide patterns that suggest database interaction"""
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            qualified_class_name = f"{file_package}.{class_name}"
            
            # Check for database-related fields
            db_field_types = [
                'Connection', 'PreparedStatement', 'Statement', 'ResultSet',
                'DataSource', 'JdbcTemplate', 'EntityManager', 'Session',
                'SessionFactory', 'SqlSession'
            ]
            
            has_db_fields = False
            for field in node.fields:
                if isinstance(field.type, javalang.tree.ReferenceType) and \
                   field.type.name in db_field_types:
                    has_db_fields = True
                    self.db_related_classes.add(qualified_class_name)
                    break
            
            # If class has DB fields, all its methods might be DB-related
            if has_db_fields:
                for method_name in self.class_methods.get(qualified_class_name, []):
                    qualified_method_name = f"{qualified_class_name}.{method_name}"
                    self.db_related_methods.add(qualified_method_name)

    def build_recursive_dependency_graph(self, max_depth=10):
        """Build a recursive dependency graph by following all method calls"""
        print("Building recursive dependency graph...")
        
        # First, filter direct dependencies to only include known methods
        for caller, callees in list(self.dependency_graph.items()):
            self.dependency_graph[caller] = {callee for callee in callees if callee in self.all_methods}
        
        # Now build the recursive graph
        for method in self.all_methods:
            self._find_recursive_dependencies(method, method, set(), 0, max_depth)
        
        print(f"Recursive analysis complete. Found dependencies for {len(self.recursive_dependencies)} methods")
        return self.recursive_dependencies

    def _find_recursive_dependencies(self, root_method, current_method, visited, current_depth, max_depth):
        """
        Recursively find all dependencies of a method
        
        Args:
            root_method: The original method we're finding dependencies for
            current_method: The current method being analyzed
            visited: Set of already visited methods in this recursion path
            current_depth: Current recursion depth
            max_depth: Maximum recursion depth
        """
        # Stop recursion if we've reached max depth or already visited this method
        if current_depth >= max_depth or current_method in visited:
            return
        
        # Mark this method as visited in this path
        visited.add(current_method)
        
        # Record the call depth for visualization
        if current_method != root_method:
            self.recursive_dependencies[root_method].add(current_method)
            
            # Track the shortest call depth to this method
            current_depth_value = self.call_depths.get((root_method, current_method), float('inf'))
            self.call_depths[(root_method, current_method)] = min(current_depth, current_depth_value)
        
        # Continue recursion for each callee
        if current_method in self.dependency_graph:
            for callee in self.dependency_graph[current_method]:
                if callee in self.all_methods:  # Ensure we only follow known methods
                    self._find_recursive_dependencies(root_method, callee, visited.copy(), current_depth + 1, max_depth)

    def analyze_codebase(self, directory_path, recursive=True, max_depth=10, deep_analysis=True):
        """Analyze the entire codebase with option for deeper analysis"""
        try:
            java_files = self.scan_directory(directory_path)
            print(f"Found {len(java_files)} Java files")
            
            file_count = 0
            for file_path in java_files:
                try:
                    file_count += 1
                    print(f"[{file_count}/{len(java_files)}] Parsing {file_path}")
                    self.parse_java_file(file_path)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
            
            # Filter dependency graph for known methods only
            for caller, callees in list(self.dependency_graph.items()):
                self.dependency_graph[caller] = {callee for callee in callees if callee in self.all_methods}
            
            print(f"Initial analysis complete. Found {len(self.dependency_graph)} direct method dependencies")
            print(f"Found {len(self.db_tables)} database tables accessed in the code")
            print(f"Found {len(self.db_related_methods)} methods that appear to be database-related")
            
            # Build recursive dependency graph if requested
            if recursive:
                self.build_recursive_dependency_graph(max_depth)
            
            # Deeper analysis to propagate database relationships (methods calling methods accessing DBs)
            if deep_analysis:
                print("Performing deep analysis of database interactions...")
                self.propagate_db_relationships(max_depth)
                
            return self.dependency_graph, self.recursive_dependencies
        except Exception as e:
            print(f"Error during codebase analysis: {str(e)}")
            return {}, defaultdict(set)

    def propagate_db_relationships(self, max_depth=5):
        """Propagate DB relationships through the call graph to find indirect DB access"""
        # Start with methods directly accessing tables
        db_accessing_methods = set(self.method_table_access.keys())
        initial_count = len(db_accessing_methods)
        
        # Find all methods that call methods that access databases
        iteration = 1
        while iteration <= max_depth:
            newly_added = set()
            
            for caller, callees in self.dependency_graph.items():
                # If caller is not already known to access DB
                if caller not in db_accessing_methods:
                    # Check if it calls any methods that do
                    for callee in callees:
                        if callee in db_accessing_methods:
                            newly_added.add(caller)
                            
                            # Mark all tables accessed by callee as indirectly accessed by caller
                            for callee_method in db_accessing_methods:
                                if callee == callee_method:
                                    for table in self.method_table_access[callee]:
                                        self.method_table_access[caller].add(table)
                                        # Mark it as an indirect access
                                        access_types = self.table_access_types.get((callee, table), {'UNKNOWN'})
                                        for access_type in access_types:
                                            self.table_access_types[(caller, table)].add(f"INDIRECT_{access_type}")
                            
                            # Also mark this method as DB-related
                            self.db_related_methods.add(caller)
            
            # Add newly discovered methods to our known set
            db_accessing_methods.update(newly_added)
            
            if not newly_added:
                break
                
            print(f"Iteration {iteration}: Found {len(newly_added)} more methods with indirect database access")
            iteration += 1
        
        print(f"Deep analysis complete. Found {len(db_accessing_methods) - initial_count} additional methods with indirect database access.")

    def export_dot(self, output_file, recursive=True, include_db=True, db_only=False):
        """Export dependency graph in DOT format"""
        with open(output_file, 'w') as file:
            file.write("digraph dependencies {\n")
            file.write("  node [shape=box, style=filled, fillcolor=lightblue];\n")
            file.write("  rankdir=LR;\n")
            
            # Add nodes for methods (skip if db_only mode)
            if not db_only:
                for method in self.all_methods:
                    # Only include methods that interact with databases if db_only mode is on
                    if db_only and method not in self.method_table_access:
                        continue
                        
                    # Escape quotes in method names
                    safe_method = method.replace('"', '\\"')
                    parts = safe_method.split('.')
                    method_name = parts[-1]
                    class_name = parts[-2]
                    package = '.'.join(parts[:-2])
                    
                    label = f"{class_name}.{method_name}\\n({package})"
                    file.write(f'  "{safe_method}" [label="{label}"];\n')
            
            # Add database tables as nodes if requested
            if include_db and self.db_tables:
                file.write("\n  # Database tables\n")
                for table in self.db_tables:
                    safe_table = table.replace('"', '\\"')
                    file.write(f'  "DB_TABLE:{safe_table}" [label="{safe_table}", shape=cylinder, '
                               f'fillcolor="#E6FFCC", style=filled];\n')
            
            # Add edges for method dependencies (skip if db_only mode)
            if not db_only:
                file.write("\n  # Method call dependencies\n")
                if recursive:
                    # Use recursive dependencies
                    for caller, callees in self.recursive_dependencies.items():
                        # Skip if db_only and the caller doesn't access any db tables
                        if db_only and caller not in self.method_table_access:
                            continue
                            
                        safe_caller = caller.replace('"', '\\"')
                        for callee in callees:
                            # Skip if db_only and the callee doesn't access any db tables
                            if db_only and callee not in self.method_table_access:
                                continue
                                
                            safe_callee = callee.replace('"', '\\"')
                            depth = self.call_depths.get((caller, callee), 1)
                            penwidth = max(1, 4 - depth * 0.5)  # Thicker lines for closer relationships
                            color = f"\"#{max(0, 255 - depth * 40):02x}00ff\""  # Color gradient based on depth
                            file.write(f'  "{safe_caller}" -> "{safe_callee}" [penwidth={penwidth}, color={color}, label="{depth}"];\n')
                else:
                    # Use direct dependencies only
                    for caller, callees in self.dependency_graph.items():
                        # Skip if db_only and the caller doesn't access any db tables
                        if db_only and caller not in self.method_table_access:
                            continue
                            
                        safe_caller = caller.replace('"', '\\"')
                        for callee in callees:
                            # Skip if db_only and the callee doesn't access any db tables
                            if db_only and callee not in self.method_table_access:
                                continue
                                
                            safe_callee = callee.replace('"', '\\"')
                            file.write(f'  "{safe_caller}" -> "{safe_callee}";\n')
            
            # Add edges for database access if requested
            if include_db and self.method_table_access:
                file.write("\n  # Database access dependencies\n")
                for method, tables in self.method_table_access.items():
                    safe_method = method.replace('"', '\\"')
                    for table in tables:
                        safe_table = table.replace('"', '\\"')
                        # Get access types for this method-table pair
                        access_types = self.table_access_types.get((method, table), {'UNKNOWN'})
                        access_label = ", ".join(sorted(access_types))
                        
                        # Different colors for different access types
                        if 'SELECT' in access_types:
                            color = "#0000FF"  # Blue for SELECT
                        elif 'INSERT' in access_types:
                            color = "#00FF00"  # Green for INSERT
                        elif 'UPDATE' in access_types:
                            color = "#FFA500"  # Orange for UPDATE
                        elif 'DELETE' in access_types:
                            color = "#FF0000"  # Red for DELETE
                        else:
                            color = "#808080"  # Gray for other/unknown
                            
                        file.write(f'  "{safe_method}" -> "DB_TABLE:{safe_table}" '
                                   f'[label="{access_label}", color="{color}", style=dashed];\n')
            
            file.write("}\n")
        
        print(f"Dependency graph written to {output_file}")

    def export_json(self, output_file, recursive=True, include_db=True, db_only=False):
        """Export dependency graph in JSON format for web-based visualization"""
        import json
        
        nodes = []
        links = []
        
        # Create a unique ID for each method and table
        method_to_id = {}
        method_index = 0
        
        # Add nodes for methods that we'll include
        for method in self.all_methods:
            # Skip methods that don't access databases if db_only is True
            if db_only and method not in self.method_table_access:
                continue
                
            parts = method.split('.')
            method_name = parts[-1]
            class_name = parts[-2]
            package = '.'.join(parts[:-2])
            
            method_to_id[method] = method_index
            method_index += 1
            
            nodes.append({
                "id": method_to_id[method],
                "name": method,
                "method": method_name,
                "class": class_name,
                "package": package,
                "type": "method",
                "db_accessor": method in self.method_table_access  # Flag if it accesses DB
            })
        
        # Start table IDs after method IDs
        table_offset = method_index
        table_to_id = {table: i + table_offset for i, table in enumerate(self.db_tables)}
        
        # Add nodes for database tables
        if include_db:
            for table in self.db_tables:
                nodes.append({
                    "id": table_to_id[table],
                    "name": table,
                    "type": "database",
                    "table": table
                })
        
        # Add links between methods (only if not db_only)
        if not db_only:
            if recursive:
                # Use recursive dependencies
                for caller, callees in self.recursive_dependencies.items():
                    if caller in method_to_id:  # Check if caller is included
                        source_id = method_to_id[caller]
                        for callee in callees:
                            if callee in method_to_id:  # Check if callee is included
                                target_id = method_to_id[callee]
                                depth = self.call_depths.get((caller, callee), 1)
                                links.append({
                                    "source": source_id,
                                    "target": target_id,
                                    "depth": depth,
                                    "value": max(1, 5 - depth),  # Stronger links for closer relationships
                                    "type": "method_call"
                                })
            else:
                # Use direct dependencies only
                for caller, callees in self.dependency_graph.items():
                    if caller in method_to_id:  # Check if caller is included
                        source_id = method_to_id[caller]
                        for callee in callees:
                            if callee in method_to_id:  # Check if callee is included
                                target_id = method_to_id[callee]
                                links.append({
                                    "source": source_id,
                                    "target": target_id,
                                    "depth": 1,
                                    "value": 4,
                                    "type": "method_call"
                                })
        
        # Add links between methods and database tables
        if include_db:
            for method, tables in self.method_table_access.items():
                if method in method_to_id:  # Check if method is included
                    source_id = method_to_id[method]
                    for table in tables:
                        if table in table_to_id:
                            target_id = table_to_id[table]
                            access_types = sorted(self.table_access_types.get((method, table), {'UNKNOWN'}))
                            links.append({
                                "source": source_id,
                                "target": target_id,
                                "access_types": list(access_types),
                                "value": 3,
                                "type": "db_access"
                            })
        
        data = {"nodes": nodes, "links": links}
        
        with open(output_file, 'w') as file:
            json.dump(data, file, indent=2)
        
        print(f"JSON dependency data written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Java codebase for method dependencies')
    parser.add_argument('directory', help='Directory containing Java files to analyze')
    parser.add_argument('-o', '--output', default='dependency_graph.dot', 
                        help='Output file path for DOT graph (default: dependency_graph.dot)')
    parser.add_argument('-j', '--json', default='dependency_graph.json',
                        help='Output file path for JSON data (default: dependency_graph.json)')
    parser.add_argument('-r', '--recursive', action='store_true', default=True,
                        help='Analyze dependencies recursively (default: True)')
    parser.add_argument('-d', '--depth', type=int, default=5,
                        help='Maximum recursion depth for dependency analysis (default: 5)')
    parser.add_argument('--direct-only', action='store_true',
                        help='Only analyze direct method calls, not recursive dependencies')
    parser.add_argument('--no-deep', action='store_true',
                        help='Skip deep analysis of database interactions')
    
    # Database-related options
    db_group = parser.add_argument_group('Database Analysis')
    db_group.add_argument('--no-db', action='store_true',
                        help='Exclude database table analysis')
    db_group.add_argument('--db-only', action='store_true',
                        help='Focus only on database interactions (methods that interact with databases)')
    
    args = parser.parse_args()
    
    # Validate conflicting options
    if args.no_db and args.db_only:
        parser.error("--no-db and --db-only cannot be used together")
    
    analyzer = JavaDependencyAnalyzer()
    analyzer.analyze_codebase(args.directory, not args.direct_only, args.depth, not args.no_deep)
    analyzer.export_dot(args.output, not args.direct_only, not args.no_db, args.db_only)
    analyzer.export_json(args.json, not args.direct_only, not args.no_db, args.db_only)
    
    # Provide appropriate message based on the mode
    mode_msg = ""
    if args.db_only:
        mode_msg = "Database interaction mode: showing only methods that access databases"
    elif args.no_db:
        mode_msg = "Method call only mode: database interactions are excluded"
    
    print(f"""
Analysis complete! {mode_msg}

To visualize the graph using Graphviz:
    dot -Tpng {args.output} -o dependency_graph.png

For web-based visualization, use the JSON file with a tool like D3.js:
    {args.json}
    """)


if __name__ == "__main__":
    main()
