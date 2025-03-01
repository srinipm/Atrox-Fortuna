#!/usr/bin/env python3
"""
Sybase to Java Converter

This script converts Sybase stored procedures to Java classes with JDBC.
It reads SQL files from an input directory, applies conversion rules,
and writes the converted Java classes to an output directory.

Usage:
    python sybase_to_java.py --input <input_dir> --output <output_dir>
"""

import os
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path

class SybaseToJavaConverter:
    """Converts Sybase stored procedures to Java classes with JDBC."""
    
    def __init__(self, logger):
        self.logger = logger
        self.java_imports = [
            "import java.sql.Connection;",
            "import java.sql.PreparedStatement;",
            "import java.sql.ResultSet;",
            "import java.sql.SQLException;",
            "import java.sql.Timestamp;",
            "import java.math.BigDecimal;",
            "import java.util.ArrayList;",
            "import java.util.Date;",
            "import java.util.List;",
            "import java.util.HashMap;",
            "import java.util.Map;",
            "import java.time.LocalDate;",
            "import java.time.LocalDateTime;"
        ]
        
    def _sql_type_to_java(self, sql_type, size=None, precision=None):
        """Convert SQL data type to appropriate Java data type."""
        sql_type = sql_type.upper() if sql_type else ""
        
        # Map of SQL types to Java types
        type_map = {
            'INTEGER': 'int',
            'INT': 'int',
            'SMALLINT': 'short',
            'TINYINT': 'byte',
            'BIGINT': 'long',
            'DECIMAL': 'BigDecimal',
            'NUMERIC': 'BigDecimal',
            'FLOAT': 'double',
            'REAL': 'float',
            'MONEY': 'BigDecimal',
            'SMALLMONEY': 'BigDecimal',
            'BIT': 'boolean',
            'CHAR': 'String',
            'VARCHAR': 'String',
            'TEXT': 'String',
            'NCHAR': 'String',
            'NVARCHAR': 'String',
            'NTEXT': 'String',
            'BINARY': 'byte[]',
            'VARBINARY': 'byte[]',
            'IMAGE': 'byte[]',
            'DATETIME': 'Timestamp',
            'SMALLDATETIME': 'Timestamp',
            'DATE': 'java.sql.Date',
            'TIME': 'java.sql.Time',
            'UNIQUEIDENTIFIER': 'java.util.UUID',
        }
        
        return type_map.get(sql_type, 'Object')
    
    def _extract_procedure_name(self, content):
        """Extract procedure name from stored procedure definition."""
        match = re.search(r'CREATE\s+PROCEDURE\s+(\w+)', content, re.IGNORECASE)
        if match:
            return match.group(1)
        return "StoredProcedure"
    
    def _extract_parameters(self, content):
        """Extract parameters from stored procedure definition."""
        # Find the parentheses after CREATE PROCEDURE name
        match = re.search(r'CREATE\s+PROCEDURE\s+\w+\s*\((.*?)\)', content, re.IGNORECASE | re.DOTALL)
        
        if not match:
            # Some procedures define parameters after the name without parentheses
            match = re.search(r'CREATE\s+PROCEDURE\s+\w+\s+((?:@\w+\s+\w+(?:\(\d+(?:,\d+)?\))?(?:\s*=\s*[^,]+)?(?:,|\s+AS|\s+BEGIN)))', 
                             content, re.IGNORECASE | re.DOTALL)
        
        if match:
            param_text = match.group(1)
            # Parse parameters in format @name type(size)
            params = []
            param_matches = re.finditer(r'@(\w+)\s+([A-Za-z]+)(?:\((\d+)(?:,\s*(\d+))?\))?(?:\s*=\s*([^,]+))?', 
                                       param_text, re.IGNORECASE)
            
            for param in param_matches:
                name = param.group(1)
                sql_type = param.group(2)
                size = param.group(3)
                precision = param.group(4)
                default = param.group(5)
                
                java_type = self._sql_type_to_java(sql_type, size, precision)
                params.append({
                    'name': name,
                    'java_type': java_type,
                    'sql_type': sql_type,
                    'default': default
                })
            
            return params
        
        return []
    
    def _extract_variables(self, content):
        """Extract variable declarations from stored procedure."""
        variables = []
        # Find all DECLARE statements
        var_matches = re.finditer(r'DECLARE\s+@(\w+)\s+([A-Za-z]+)(?:\((\d+)(?:,\s*(\d+))?\))?', 
                                content, re.IGNORECASE)
        
        for var in var_matches:
            name = var.group(1)
            sql_type = var.group(2)
            size = var.group(3)
            precision = var.group(4)
            
            java_type = self._sql_type_to_java(sql_type, size, precision)
            variables.append({
                'name': name,
                'java_type': java_type,
                'sql_type': sql_type
            })
        
        return variables
    
    def _extract_sql_statements(self, content):
        """Extract SQL statements from stored procedure."""
        # Remove comments to avoid false matches
        content_no_comments = re.sub(r'--.*$', '', content, flags=re.MULTILINE)
        content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)
        
        # Define patterns for different SQL statements
        patterns = {
            'select': r'SELECT\s+(.*?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?(?:\s+ORDER\s+BY\s+(.+?))?(?:\s+GROUP\s+BY\s+(.+?))?(?:$|;|\s+GO\s+)',
            'insert': r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)',
            'update': r'UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+?))?(?:$|;|\s+GO\s+)',
            'delete': r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?(?:$|;|\s+GO\s+)',
            'if': r'IF\s+(.+?)\s+BEGIN\s+(.*?)\s+END',
            'while': r'WHILE\s+(.+?)\s+BEGIN\s+(.*?)\s+END',
        }
        
        statements = []
        for stmt_type, pattern in patterns.items():
            for match in re.finditer(pattern, content_no_comments, re.IGNORECASE | re.DOTALL):
                statements.append({
                    'type': stmt_type,
                    'match': match
                })
        
        return sorted(statements, key=lambda s: s['match'].start())
    
    def _generate_java_for_select(self, match):
        """Generate Java code for a SELECT statement."""
        columns = match.group(1).strip()
        table = match.group(2).strip()
        where_clause = match.group(3)
        order_by = match.group(4)
        
        java_code = []
        
        # Build SQL string
        sql = f'String sql = "SELECT {columns} FROM {table}'
        
        if where_clause:
            sql += f" WHERE {where_clause}"
            
        if order_by:
            sql += f" ORDER BY {order_by}"
            
        sql += '";'
        java_code.append(sql)
        
        # Java implementation using JDBC
        java_code.append("List<Map<String, Object>> resultList = new ArrayList<>();")
        java_code.append("try (PreparedStatement stmt = connection.prepareStatement(sql)) {")
        java_code.append("    // TODO: Set any parameters for prepared statement")
        java_code.append("    try (ResultSet rs = stmt.executeQuery()) {")
        java_code.append("        while (rs.next()) {")
        java_code.append("            Map<String, Object> row = new HashMap<>();")
        
        # Extract column names for result processing
        column_names = [c.strip().split(' ')[-1].split('.')[-1] for c in columns.split(',')]
        for col in column_names:
            if col == '*':
                java_code.append("            // Handle all columns in result set")
                java_code.append("            for (int i = 1; i <= rs.getMetaData().getColumnCount(); i++) {")
                java_code.append("                String colName = rs.getMetaData().getColumnName(i);")
                java_code.append("                row.put(colName, rs.getObject(i));")
                java_code.append("            }")
            else:
                java_code.append(f"            row.put(\"{col}\", rs.getObject(\"{col}\"));")
                
        java_code.append("            resultList.add(row);")
        java_code.append("        }")
        java_code.append("    }")
        java_code.append("}")
        java_code.append("return resultList;")
        
        return "\n".join(java_code)
    
    def _generate_java_for_insert(self, match):
        """Generate Java code for an INSERT statement."""
        table = match.group(1).strip()
        columns = [c.strip() for c in match.group(2).split(',')]
        values = [v.strip() for v in match.group(3).split(',')]
        
        java_code = []
        
        # Build SQL string with parameterized query
        sql = f'String sql = "INSERT INTO {table} ({", ".join(columns)}) VALUES ({", ".join(["?"] * len(columns))})";'
        java_code.append(sql)
        
        # Java implementation using JDBC
        java_code.append("try (PreparedStatement stmt = connection.prepareStatement(sql)) {")
        
        # Set parameter values
        for i, (col, val) in enumerate(zip(columns, values)):
            java_code.append(f"    // TODO: Properly set parameter type for '{col}'")
            java_code.append(f"    stmt.setObject({i+1}, {val});  // Parameter index is 1-based")
        
        java_code.append("    int rowsAffected = stmt.executeUpdate();")
        java_code.append("    return rowsAffected;")
        java_code.append("}")
        
        return "\n".join(java_code)
    
    def _generate_java_for_update(self, match):
        """Generate Java code for an UPDATE statement."""
        table = match.group(1).strip()
        set_clause = match.group(2).strip()
        where_clause = match.group(3)
        
        java_code = []
        
        # Build SQL string
        sql = f'String sql = "UPDATE {table} SET {set_clause}'
        
        if where_clause:
            sql += f" WHERE {where_clause}"
            
        sql += '";'
        java_code.append(sql)
        
        # Java implementation using JDBC
        java_code.append("try (PreparedStatement stmt = connection.prepareStatement(sql)) {")
        java_code.append("    // TODO: Set parameters for prepared statement")
        java_code.append("    int rowsAffected = stmt.executeUpdate();")
        java_code.append("    return rowsAffected;")
        java_code.append("}")
        
        return "\n".join(java_code)
    
    def _generate_java_for_delete(self, match):
        """Generate Java code for a DELETE statement."""
        table = match.group(1).strip()
        where_clause = match.group(2)
        
        java_code = []
        
        # Build SQL string
        sql = f'String sql = "DELETE FROM {table}'
        
        if where_clause:
            sql += f" WHERE {where_clause}"
            
        sql += '";'
        java_code.append(sql)
        
        # Java implementation using JDBC
        java_code.append("try (PreparedStatement stmt = connection.prepareStatement(sql)) {")
        java_code.append("    // TODO: Set parameters for prepared statement")
        java_code.append("    int rowsAffected = stmt.executeUpdate();")
        java_code.append("    return rowsAffected;")
        java_code.append("}")
        
        return "\n".join(java_code)
    
    def _generate_java_class(self, content):
        """Generate Java class from Sybase stored procedure."""
        proc_name = self._extract_procedure_name(content)
        parameters = self._extract_parameters(content)
        variables = self._extract_variables(content)
        
        # Start building the Java class
        java_lines = []
        
        # Package declaration - can be customized
        java_lines.append("package com.example.procedures;")
        java_lines.append("")
        
        # Add imports
        java_lines.extend(self.java_imports)
        java_lines.append("")
        
        # Class definition
        java_lines.append(f"/**")
        java_lines.append(f" * Java implementation of Sybase stored procedure {proc_name}")
        java_lines.append(f" * Generated by SybaseToJavaConverter")
        java_lines.append(f" */")
        java_lines.append(f"public class {proc_name} {{")
        
        # Add connection field
        java_lines.append("    private final Connection connection;")
        java_lines.append("")
        
        # Add constructor
        java_lines.append(f"    /**")
        java_lines.append(f"     * Creates a new {proc_name} instance")
        java_lines.append(f"     * @param connection JDBC connection to use")
        java_lines.append(f"     */")
        java_lines.append(f"    public {proc_name}(Connection connection) {{")
        java_lines.append(f"        this.connection = connection;")
        java_lines.append(f"    }}")
        java_lines.append("")
        
        # Add execute method with parameters
        java_lines.append(f"    /**")
        java_lines.append(f"     * Executes the stored procedure")
        for param in parameters:
            java_lines.append(f"     * @param {param['name']} Parameter description")
        java_lines.append(f"     * @return Result as List of Maps")
        java_lines.append(f"     * @throws SQLException if a database error occurs")
        java_lines.append(f"     */")
        
        # Method signature with parameters
        param_str = ", ".join([f"{p['java_type']} {p['name']}" for p in parameters])
        java_lines.append(f"    public List<Map<String, Object>> execute({param_str}) throws SQLException {{")
        
        # Add variable declarations
        if variables:
            java_lines.append("        // Variable declarations")
            for var in variables:
                default_value = self._get_default_value(var['java_type'])
                java_lines.append(f"        {var['java_type']} {var['name']} = {default_value};")
            java_lines.append("")
        
        # Add comment for manual implementation
        java_lines.append("        // TODO: Implement procedure logic")
        java_lines.append("        List<Map<String, Object>> resultList = new ArrayList<>();")
        java_lines.append("")
        
        # Sample implementation based on procedure's SQL statements
        java_lines.append("        try {")
        java_lines.append("            // Begin transaction if needed")
        java_lines.append("            boolean originalAutoCommit = connection.getAutoCommit();")
        java_lines.append("            connection.setAutoCommit(false);")
        java_lines.append("")
        java_lines.append("            // Placeholder for procedure implementation")
        java_lines.append("            // The actual SQL operations should be converted here")
        java_lines.append("")
        java_lines.append("            // Commit transaction")
        java_lines.append("            connection.commit();")
        java_lines.append("            connection.setAutoCommit(originalAutoCommit);")
        java_lines.append("        } catch (SQLException e) {")
        java_lines.append("            // Rollback transaction on error")
        java_lines.append("            try {")
        java_lines.append("                connection.rollback();")
        java_lines.append("            } catch (SQLException rollbackEx) {")
        java_lines.append("                e.addSuppressed(rollbackEx);")
        java_lines.append("            }")
        java_lines.append("            throw e;")
        java_lines.append("        }")
        java_lines.append("")
        
        # Return statement
        java_lines.append("        return resultList;")
        java_lines.append("    }")
        
        # Add helper methods if needed
        java_lines.append("")
        java_lines.append("    // Helper methods for procedure implementation")
        
        # Close class
        java_lines.append("}")
        
        return "\n".join(java_lines)
    
    def _get_default_value(self, java_type):
        """Get default Java value for a given type."""
        defaults = {
            'int': '0',
            'long': '0L',
            'short': '(short)0',
            'byte': '(byte)0',
            'double': '0.0',
            'float': '0.0f',
            'boolean': 'false',
            'BigDecimal': 'null',
            'String': 'null',
            'byte[]': 'null',
            'Timestamp': 'null',
            'java.sql.Date': 'null',
            'java.sql.Time': 'null',
            'java.util.UUID': 'null'
        }
        
        return defaults.get(java_type, 'null')
    
    def convert_file(self, input_file, output_file):
        """
        Converts a single Sybase stored procedure file to Java code.
        
        Args:
            input_file: Path to the input Sybase procedure file
            output_file: Path to write the converted Java class
            
        Returns:
            bool: True if conversion was successful
        """
        try:
            self.logger.info(f"Converting {input_file} to Java")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract procedure name for class name and file name
            proc_name = self._extract_procedure_name(content)
            if not proc_name:
                self.logger.warning(f"Could not extract procedure name from {input_file}")
                proc_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # Generate Java source code
            java_code = self._generate_java_class(content)
            
            # Update output file path with class name
            java_file = output_file.with_name(proc_name + ".java")
            
            # Write Java class
            with open(java_file, 'w', encoding='utf-8') as f:
                f.write(java_code)
                
            self.logger.info(f"Generated Java class {proc_name} in {java_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting {input_file}: {str(e)}")
            return False
    
    def process_directory(self, input_dir, output_dir):
        """
        Processes all files in the input directory and converts them to Java code.
        
        Args:
            input_dir: Directory containing Sybase stored procedure files
            output_dir: Directory to write converted Java classes
            
        Returns:
            tuple: (files_processed, files_converted)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Make sure output directory exists
        output_path.mkdir(exist_ok=True, parents=True)
        
        files_processed = 0
        files_converted = 0
        
        self.logger.info(f"Starting conversion from {input_dir} to {output_dir}")
        
        # Process all .sql files in input directory
        for file_path in input_path.glob('**/*.sql'):
            # Determine the relative path to maintain directory structure
            rel_path = file_path.relative_to(input_path)
            output_file = output_path / rel_path
            
            # Create subdirectories if needed
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Convert the file
            files_processed += 1
            if self.convert_file(file_path, output_file):
                files_converted += 1
            
        self.logger.info(f"Conversion complete. Processed {files_processed} files, successfully converted {files_converted}.")
        return (files_processed, files_converted)


def setup_logging():
    """
    Sets up logging to both console and file.
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('sybase_to_java')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler with timestamped log file
    log_file = f"java_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(description='Convert Sybase stored procedures to Java classes with JDBC.')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing Sybase stored procedures')
    parser.add_argument('--output', '-o', required=True, help='Output directory for converted Java classes')
    parser.add_argument('--package', '-p', default='com.example.procedures', help='Java package name for generated classes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    logger.info(f"Starting Sybase to Java conversion")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Java package: {args.package}")
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        logger.error(f"Input directory {args.input} does not exist or is not a directory")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logger.info(f"Created output directory: {args.output}")
    
    # Initialize converter
    converter = SybaseToJavaConverter(logger)
    
    # Process the directory
    files_processed, files_converted = converter.process_directory(args.input, args.output)
    
    logger.info("Conversion completed")
    logger.info(f"Files processed: {files_processed}")
    logger.info(f"Files successfully converted: {files_converted}")
    
    return 0


if __name__ == "__main__":
    exit(main())
