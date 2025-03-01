#!/usr/bin/env python3
"""
Enhanced Sybase to Modern Database Converter

This tool converts C++ code using Sybase CT-Lib API to equivalent code 
for MS SQL Server, Oracle, or MongoDB.

Usage: python sybase_converter.py --input-dir /path/to/source --target [mssql|oracle|mongodb] [--backup-dir /path/to/backups]
"""

import re
import os
import sys
import argparse
import shutil
import glob
import logging
from enum import Enum
from datetime import datetime


class TargetDatabase(Enum):
    MSSQL = "mssql"
    ORACLE = "oracle"
    MONGODB = "mongodb"


class SybaseConverter:
    def __init__(self, target_db, log_file):
        self.target_db = target_db
        self.logger = self._setup_logger(log_file)
        self.include_replacements = {
            TargetDatabase.MSSQL: "#include <sqlncli.h>\n#include <sqlext.h>",
            TargetDatabase.ORACLE: "#include <occi.h>",
            TargetDatabase.MONGODB: "#include <mongocxx/client.hpp>\n#include <mongocxx/instance.hpp>\n#include <bsoncxx/builder/stream/document.hpp>"
        }
        
        self.namespace_additions = {
            TargetDatabase.MSSQL: "",
            TargetDatabase.ORACLE: "using namespace oracle::occi;",
            TargetDatabase.MONGODB: "using namespace mongocxx;"
        }
        
        # Initialize variable mappings for each target database
        self.init_variable_mappings()
        self.init_function_mappings()
        
        # Track changes made to files
        self.changes_made = 0
    
    def _setup_logger(self, log_file):
        """Set up file logger"""
        logger = logging.getLogger('sybase_converter')
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def init_variable_mappings(self):
        """Initialize mappings for variables and types"""
        # Same as before...
        self.variable_mappings = {
            TargetDatabase.MSSQL: {
                "CS_CONTEXT": "SQLHENV",
                "CS_CONNECTION": "SQLHDBC", 
                "CS_COMMAND": "SQLHSTMT",
                "CS_RETCODE": "SQLRETURN",
                "CS_INT": "SQLINTEGER",
                "CS_CHAR": "SQLCHAR",
                "CS_FLOAT": "SQLFLOAT",
                "CS_VOID": "void",
                "CS_SUCCEED": "SQL_SUCCESS",
                "CS_UNUSED": "SQL_NULL_HANDLE",
                "CS_DATAFMT": "SQLSMALLINT",
                "CS_NULLTERM": "SQL_NTS"
            },
            TargetDatabase.ORACLE: {
                "CS_CONTEXT": "Environment*",
                "CS_CONNECTION": "Connection*",
                "CS_COMMAND": "Statement*",
                "CS_RETCODE": "Status",
                "CS_INT": "int",
                "CS_CHAR": "char",
                "CS_FLOAT": "float",
                "CS_VOID": "void",
                "CS_SUCCEED": "SUCCESS",
                "CS_UNUSED": "0",
                "CS_DATAFMT": "Type",
                "CS_NULLTERM": "-1"
            },
            TargetDatabase.MONGODB: {
                "CS_CONTEXT": "mongocxx::instance",
                "CS_CONNECTION": "mongocxx::client",
                "CS_COMMAND": "mongocxx::collection",
                "CS_RETCODE": "bool",
                "CS_INT": "int32_t",
                "CS_CHAR": "std::string",
                "CS_FLOAT": "double",
                "CS_VOID": "void",
                "CS_SUCCEED": "true",
                "CS_UNUSED": "0",
                "CS_DATAFMT": "bsoncxx::document::view",
                "CS_NULLTERM": "0"
            }
        }
        
    def init_function_mappings(self):
        """Initialize mappings for Sybase functions to target database functions"""
        # Same function mappings as before...
        self.mssql_function_mappings = {
            r'cs_ctx_alloc\s*\(\s*CS_VERSION_\w+\s*,\s*&([^)]+)\)': 
                r'SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &\1)',
            r'ct_init\s*\(\s*([^,]+)\s*,\s*CS_VERSION_\w+\s*\)': 
                r'SQLSetEnvAttr(\1, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0)',
            r'ct_con_alloc\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'SQLAllocHandle(SQL_HANDLE_DBC, \1, &\2)',
            r'ct_connect\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*CS_NULLTERM\s*\)': 
                r'SQLDriverConnect(\1, NULL, (SQLCHAR*)"SERVER=\2;Trusted_Connection=yes;", SQL_NTS, NULL, 0, NULL, SQL_DRIVER_NOPROMPT)',
            r'ct_cmd_alloc\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'SQLAllocHandle(SQL_HANDLE_STMT, \1, &\2)',
            r'ct_command\s*\(\s*([^,]+)\s*,\s*CS_LANG_CMD\s*,\s*([^,]+)\s*,\s*CS_NULLTERM\s*,\s*CS_UNUSED\s*\)': 
                r'SQLPrepare(\1, (SQLCHAR*)\2, SQL_NTS)',
            r'ct_send\s*\(\s*([^)]+)\)': 
                r'SQLExecute(\1)',
            r'ct_results\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'SQLNumResultCols(\1, &\2) == SQL_SUCCESS ? SQL_SUCCESS : SQL_NO_DATA',
            r'ct_fetch\s*\(\s*([^,]+).*?\)': 
                r'SQLFetch(\1)',
            r'ct_close\s*\((\s*([^,]+)\s*,\s*CS_UNUSED\s*)': 
                r'SQLDisconnect(\1)',
            r'ct_con_drop\s*\(\s*([^)]+)\)': 
                r'SQLFreeHandle(SQL_HANDLE_DBC, \1)',
            r'ct_exit\s*\(\s*([^,]+)\s*,\s*CS_UNUSED\s*\)': 
                r'SQLFreeHandle(SQL_HANDLE_ENV, \1)',
            r'cs_ctx_drop\s*\(\s*([^)]+)\)': 
                r'1 /* cs_ctx_drop not needed in ODBC */'
        }
        
        # Other function mappings remain the same...
        self.oracle_function_mappings = {
            r'cs_ctx_alloc\s*\(\s*CS_VERSION_\w+\s*,\s*&([^)]+)\)': 
                r'\1 = Environment::createEnvironment()',
            r'ct_init\s*\(\s*([^,]+)\s*,\s*CS_VERSION_\w+\s*\)': 
                r'true /* Oracle environment already initialized */',
            r'ct_con_alloc\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'\2 = \1->createConnection(username, password, connectionString)',
            r'ct_connect\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*CS_NULLTERM\s*\)': 
                r'true /* Oracle connection already established */',
            r'ct_cmd_alloc\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'\2 = \1->createStatement()',
            r'ct_command\s*\(\s*([^,]+)\s*,\s*CS_LANG_CMD\s*,\s*([^,]+)\s*,\s*CS_NULLTERM\s*,\s*CS_UNUSED\s*\)': 
                r'\1->setSQL(\2)',
            r'ct_send\s*\(\s*([^)]+)\)': 
                r'\1->execute()',
            r'ct_results\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'\2 = (\1->getResultSet() != NULL)',
            r'ct_fetch\s*\(\s*([^,]+).*?\)': 
                r'\1->getResultSet()->next()',
            r'ct_close\s*\((\s*([^,]+)\s*,\s*CS_UNUSED\s*)': 
                r'\1->terminateStatement()',
            r'ct_con_drop\s*\(\s*([^)]+)\)': 
                r'Environment::terminateConnection(\1)',
            r'ct_exit\s*\(\s*([^,]+)\s*,\s*CS_UNUSED\s*\)': 
                r'Environment::terminateEnvironment(\1)',
            r'cs_ctx_drop\s*\(\s*([^)]+)\)': 
                r'1 /* Environment termination handled by terminateEnvironment */'
        }
        
        self.mongodb_function_mappings = {
            r'cs_ctx_alloc\s*\(\s*CS_VERSION_\w+\s*,\s*&([^)]+)\)': 
                r'\1 = mongocxx::instance{}',
            r'ct_init\s*\(\s*([^,]+)\s*,\s*CS_VERSION_\w+\s*\)': 
                r'true /* MongoDB instance already initialized */',
            r'ct_con_alloc\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'\2 = mongocxx::client{mongocxx::uri{}}',
            r'ct_connect\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*CS_NULLTERM\s*\)': 
                r'true /* MongoDB connection created during client construction */',
            r'ct_cmd_alloc\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'\2 = \1[m_config.database]["customers"]',
            r'ct_command\s*\(\s*([^,]+)\s*,\s*CS_LANG_CMD\s*,\s*([^,]+)\s*,\s*CS_NULLTERM\s*,\s*CS_UNUSED\s*\)': 
                r'true /* MongoDB uses BSON documents for queries */',
            r'ct_send\s*\(\s*([^)]+)\)': 
                r'\1.find(query)',
            r'ct_results\s*\(\s*([^,]+)\s*,\s*&([^)]+)\)': 
                r'\2 = cursor.begin() != cursor.end() ? true : false',
            r'ct_fetch\s*\(\s*([^,]+).*?\)': 
                r'document = *cursor.begin(); cursor++',
            r'ct_close\s*\((\s*([^,]+)\s*,\s*CS_UNUSED\s*)': 
                r'true /* MongoDB handles connection pooling automatically */',
            r'ct_con_drop\s*\(\s*([^)]+)\)': 
                r'true /* MongoDB client will be cleaned up automatically */',
            r'ct_exit\s*\(\s*([^,]+)\s*,\s*CS_UNUSED\s*\)': 
                r'true /* MongoDB instance cleanup handled by destructor */',
            r'cs_ctx_drop\s*\(\s*([^)]+)\)': 
                r'true /* MongoDB instance cleanup handled by destructor */'
        }
        
        # Combine function mappings in one dictionary
        self.function_mappings = {
            TargetDatabase.MSSQL: self.mssql_function_mappings,
            TargetDatabase.ORACLE: self.oracle_function_mappings,
            TargetDatabase.MONGODB: self.mongodb_function_mappings
        }
        
    def convert_includes(self, content, filename):
        """Replace Sybase includes with target database includes"""
        new_includes = self.include_replacements[self.target_db]
        old_content = content
        content = re.sub(r'#include\s+"SybaseDatabaseManager\.h"', 
                         f'#include "DatabaseManager.h"\n{new_includes}', content)
        
        if old_content != content:
            self.logger.info(f"{filename}: Replaced includes with {self.target_db.value} includes")
            self.changes_made += 1
            
        return content
    
    def add_namespace(self, content, filename):
        """Add target database namespace if needed"""
        namespace_addition = self.namespace_additions[self.target_db]
        old_content = content
        
        if namespace_addition:
            # Find appropriate place to add namespace (after includes, before namespace declaration)
            match = re.search(r'(#include\s+.*\n)\s*namespace', content, re.DOTALL)
            if match:
                content = content.replace(match.group(1), 
                                        f"{match.group(1)}\n{namespace_addition}\n\n")
                
        if old_content != content:
            self.logger.info(f"{filename}: Added namespace {namespace_addition}")
            self.changes_made += 1
            
        return content
    
    def convert_class_name(self, content, filename):
        """Rename the Sybase class to appropriate target database class"""
        target_class_name = {
            TargetDatabase.MSSQL: "MSSQLDatabaseManager",
            TargetDatabase.ORACLE: "OracleDatabaseManager",
            TargetDatabase.MONGODB: "MongoDBDatabaseManager"
        }[self.target_db]
        
        old_content = content
        content = re.sub(r'SybaseDatabaseManager', target_class_name, content)
        
        if old_content != content:
            self.logger.info(f"{filename}: Renamed class from SybaseDatabaseManager to {target_class_name}")
            self.changes_made += 1
            
        return content
    
    def convert_variables(self, content, filename):
        """Convert Sybase variable types to target database variable types"""
        old_content = content
        changes = 0
        
        for sybase_type, target_type in self.variable_mappings[self.target_db].items():
            pattern = r'\b' + sybase_type + r'\b'
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, target_type, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} occurrences of {sybase_type} with {target_type}")
        
        if changes > 0:
            self.changes_made += changes
            
        return content
    
    def convert_functions(self, content, filename):
        """Convert Sybase function calls to target database function calls and add comments"""
        old_content = content
        
        for sybase_pattern, target_replacement in self.function_mappings[self.target_db].items():
            # Find all matches to add comments around them
            matches = list(re.finditer(sybase_pattern, content))
            
            # Process matches in reverse to avoid offset issues
            for match in reversed(matches):
                # Get the original text and its replacement
                original_text = content[match.start():match.end()]
                replaced_text = re.sub(sybase_pattern, target_replacement, original_text)
                
                # Add comments around the replacement
                commented_replacement = f"/* CONVERTED FROM SYBASE: {original_text.strip()} */\n{replaced_text}\n/* END CONVERSION */"
                
                # Replace in content
                content = content[:match.start()] + commented_replacement + content[match.end():]
                
                # Log the change
                self.logger.info(f"{filename}: Converted '{original_text.strip()}' to '{replaced_text.strip()}'")
                self.changes_made += 1
        
        return content
    
    def convert_error_handling(self, content, filename):
        """Convert Sybase error handling to target database error handling"""
        old_content = content
        changes = 0
        
        if self.target_db == TargetDatabase.MSSQL:
            matches = len(re.findall(r'serverMessageHandler', content))
            content = re.sub(r'serverMessageHandler', r'handleSQLServerError', content)
            if matches > 0:
                changes += matches
                self.logger.info(f"{filename}: Replaced {matches} serverMessageHandler with handleSQLServerError")
            
            matches = len(re.findall(r'clientMessageHandler', content))
            content = re.sub(r'clientMessageHandler', r'handleSQLClientError', content)
            if matches > 0:
                changes += matches
                self.logger.info(f"{filename}: Replaced {matches} clientMessageHandler with handleSQLClientError")
                
        elif self.target_db == TargetDatabase.ORACLE:
            matches = len(re.findall(r'serverMessageHandler|clientMessageHandler', content))
            content = re.sub(r'serverMessageHandler|clientMessageHandler', r'handleOracleError', content)
            if matches > 0:
                changes += matches
                self.logger.info(f"{filename}: Replaced {matches} message handlers with handleOracleError")
                
        elif self.target_db == TargetDatabase.MONGODB:
            matches = len(re.findall(r'serverMessageHandler|clientMessageHandler', content))
            content = re.sub(r'serverMessageHandler|clientMessageHandler', r'handleMongoDBError', content)
            if matches > 0:
                changes += matches
                self.logger.info(f"{filename}: Replaced {matches} message handlers with handleMongoDBError")
                
        if changes > 0:
            self.changes_made += changes
            
        return content
    
    def convert(self, input_content, filename):
        """Convert Sybase code to target database code"""
        result = input_content
        
        # Apply transformations in sequence
        result = self.convert_includes(result, filename)
        result = self.add_namespace(result, filename)
        result = self.convert_class_name(result, filename)
        result = self.convert_variables(result, filename)
        result = self.convert_functions(result, filename)
        result = self.convert_error_handling(result, filename)
        #result = self.convert_stored_procedures(result, filename)  # Add this line
        
        # Add header comment
        target_name = {
            TargetDatabase.MSSQL: "Microsoft SQL Server",
            TargetDatabase.ORACLE: "Oracle",
            TargetDatabase.MONGODB: "MongoDB"
        }[self.target_db]
        
        header = f"""/**
 * @file {os.path.basename(filename)}
 * @brief Implements secure CRUD operations for {target_name} database access
 * 
 * Converted from Sybase code by Sybase to Modern Database Converter
 * Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

"""
        result = header + result
        return result

def find_source_files(directory, include_sql=True):
    """Find all C++ and SQL source files in a directory and its subdirectories"""
    source_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.cpp', '.cxx', '.cc', '.c++', '.C')):
                source_files.append((file, os.path.join(root, file), 'cpp'))
            elif include_sql and file.endswith(('.sql', '.SQL')):
                source_files.append((file, os.path.join(root, file), 'sql'))
    return source_files

def process_sql_file(self, content, filename):
    """Process SQL file content specifically for stored procedures"""
    self.logger.info(f"Processing SQL file: {filename}")
    
    # Check if this SQL file is likely to contain stored procedures
    if not re.search(r'CREATE\s+PROC(?:EDURE)?', content, re.IGNORECASE):
        self.logger.info(f"{filename}: No stored procedures found")
        return content
        
    # Convert stored procedures if found
    result = self.convert_stored_procedures(content, filename)
    
    # Add header comment for converted SQL file
    target_name = {
        TargetDatabase.MSSQL: "Microsoft SQL Server",
        TargetDatabase.ORACLE: "Oracle",
        TargetDatabase.MONGODB: "MongoDB"
    }[self.target_db]
    
    header = f"""/*
 * File: {os.path.basename(filename)}
 * Target: {target_name}
 * 
 * Converted from Sybase stored procedures by Sybase to Modern Database Converter
 * Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

"""
    return header + result

def find_cpp_files(directory):
    """Find all C++ source files in a directory and its subdirectories"""
    cpp_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.cpp', '.cxx', '.cc', '.c++', '.C')):
                cpp_files.append(os.path.join(root, file))
    return cpp_files


def backup_file(file_path, backup_dir):
    """Create a backup of the file"""
    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Determine backup filename with directory structure preserved
    rel_path = os.path.relpath(file_path, os.path.dirname(backup_dir))
    backup_path = os.path.join(backup_dir, rel_path)
    
    # Create necessary subdirectories
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    # Copy the file
    shutil.copy2(file_path, backup_path)
    
    return backup_path


def main():
    parser = argparse.ArgumentParser(description='Convert Sybase C++ code and SQL stored procedures to other database APIs')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing C++ and SQL files with Sybase code')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for converted files')
    parser.add_argument('--backup-dir', '-b', default='backups', help='Directory to store file backups (default: ./backups)')
    parser.add_argument('--target', '-t', required=True, choices=['mssql', 'oracle', 'mongodb'],
                        help='Target database system')
    parser.add_argument('--skip-sql', action='store_true', help='Skip processing of SQL files')
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found or is not a directory")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"conversion_{args.target}_{timestamp}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Find all source files
    source_files = find_source_files(args.input_dir, not args.skip_sql)
    if not source_files:
        print(f"No source files found in '{args.input_dir}'")
        return 1
    
    cpp_files = [f for f in source_files if f[2] == 'cpp']
    sql_files = [f for f in source_files if f[2] == 'sql']
    
    print(f"Found {len(cpp_files)} C++ files and {len(sql_files)} SQL files to process")
    
    # Initialize converter
    target_db = TargetDatabase(args.target)
    converter = SybaseConverter(target_db, log_file)
    
    # Process each file
    files_processed = 0
    files_changed = 0
    backup_dir = os.path.abspath(args.backup_dir)
    
    # Process all source files
    for filename, file_path, file_type in source_files:
        try:
            # Log processing
            rel_path = os.path.relpath(file_path, args.input_dir)
            print(f"Processing {rel_path}...")
            converter.logger.info(f"Processing file: {file_path}")
            
            # Read input file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                input_content = f.read()
            
            # Save initial changes counter
            initial_changes = converter.changes_made
            
            # Process based on file type
            if file_type == 'cpp':
                output_content = converter.convert(input_content, file_path)
            else:  # SQL file
                output_content = converter.process_sql_file(input_content, file_path)
            
            files_processed += 1
            
            # Create backup before modifying anything
            backup_path = backup_file(file_path, backup_dir)
            converter.logger.info(f"Created backup at: {backup_path}")
            
            # Always write to output directory, preserving directory structure
            rel_path = os.path.relpath(file_path, args.input_dir)
            output_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write converted content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            # Check if file was actually changed
            if converter.changes_made > initial_changes:
                files_changed += 1
                converter.logger.info(f"Successfully converted: {file_path}")
                print(f"  - Modified {file_type.upper()} file with {converter.changes_made - initial_changes} changes")
            else:
                print(f"  - No changes needed but file copied to output directory")
                converter.logger.info(f"No changes made to: {file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            converter.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
    
    # Summary including SQL files
    print(f"\nConversion complete:")
    print(f"  - Processed {files_processed} files ({len(cpp_files)} C++ files, {len(sql_files)} SQL files)")
    print(f"  - Modified {files_changed} files")
    print(f"  - Made {converter.changes_made} total changes")
    print(f"  - Original files left untouched")
    print(f"  - Converted files saved to {os.path.abspath(args.output_dir)}")
    print(f"  - Backups stored in {backup_dir}")
    print(f"  - Detailed log saved to {log_file}")
    
    return 0

def test_regex_patterns(patterns_dict):
    """Test each regex pattern to find the problematic one"""
    for pattern in patterns_dict:
        try:
            re.compile(pattern)
            print(f"Pattern OK: {pattern[:50]}...")
        except re.error as e:
            print(f"ERROR in pattern: {pattern[:50]}...")
            print(f"Error message: {e}")
            return False
    return True

# Add these new methods to the SybaseConverter class

def init_stored_procedure_mappings(self):
    """Initialize mappings for Sybase stored procedure syntax to target database syntax"""
    # Common Sybase stored procedure patterns and their replacements
    
    # MS SQL Server stored procedure mappings (mostly compatible with minor changes)
    self.mssql_sp_mappings = {
        # Procedure declaration
        r'CREATE\s+PROC(?:EDURE)?\s+([^\s(]+)': r'CREATE PROCEDURE \1',
        
        # Output parameter syntax
        r'@([^\s]+)\s+([^\s]+)\s+OUTPUT': r'@\1 \2 OUTPUT',
        
        # Sybase specific functions
        r'GETDATE\(\)': r'GETDATE()',
        r'CONVERT\s*\(\s*([^,]+),\s*([^)]+)\)': r'CAST(\2 AS \1)',
        
        # Error handling
        r'RAISERROR\s*\(([^,)]+)(?:,\s*([^,)]+)(?:,\s*([^,)]+))?\)': r'RAISERROR(\1, \2, \3)',
        
        # Transaction management
        r'BEGIN\s+TRAN(?:SACTION)?': r'BEGIN TRANSACTION',
        r'COMMIT\s+TRAN(?:SACTION)?': r'COMMIT TRANSACTION',
        r'ROLLBACK\s+TRAN(?:SACTION)?': r'ROLLBACK TRANSACTION'
    }
    
    # Oracle stored procedure mappings
    self.oracle_sp_mappings = {
        # Procedure declaration
        r'CREATE\s+PROC(?:EDURE)?\s+([^\s(]+)\s*\((.*?)\)': r'CREATE OR REPLACE PROCEDURE \1(\2)',
        
        # Parameter syntax conversion
        r'@([^\s]+)\s+([^\s]+)(?:\s+OUTPUT)?': r'\1 IN OUT \2',
        r'@([^\s]+)\s+([^\s]+)': r'\1 IN \2',
        
        # Variable declarations
        r'DECLARE\s+@([^\s]+)\s+([^\s]+)': r'\1 \2;',
        
        # Control flow
        r'IF\s+(.*?)\s+BEGIN': r'IF \1 THEN',
        r'END': r'END IF;',
        r'WHILE\s+(.*?)\s+BEGIN': r'WHILE \1 LOOP',
        r'END': r'END LOOP;',
        
        # Functions
        r'GETDATE\(\)': r'SYSDATE',
        r'CONVERT\s*\(\s*([^,]+),\s*([^)]+)\)': r'CAST(\2 AS \1)',
        r'ISNULL\s*\(([^,]+),\s*([^)]+)\)': r'NVL(\1, \2)',
        
        # Error handling
        r'RAISERROR\s*\(([^,)]+)(?:,\s*([^,)]+)(?:,\s*([^,)]+))?\)': 
            r'RAISE_APPLICATION_ERROR(-20000, \1)',
        
        # Transaction management
        r'BEGIN\s+TRAN(?:SACTION)?': r'-- Transaction begins automatically',
        r'COMMIT\s+TRAN(?:SACTION)?': r'COMMIT;',
        r'ROLLBACK\s+TRAN(?:SACTION)?': r'ROLLBACK;'
    }
    
    # MongoDB stored procedure mappings (convert to JavaScript functions)
    self.mongodb_sp_mappings = {
        # Procedure declaration - convert to JavaScript function
        r'CREATE\s+PROC(?:EDURE)?\s+([^\s(]+)\s*\((.*?)\)': r'function \1(\2) {',
        
        # Parameter handling - no conversion needed, JS uses parameters directly
        r'@([^\s]+)\s+([^\s]+)(?:\s+OUTPUT)?': r'\1',
        
        # Variable declarations
        r'DECLARE\s+@([^\s]+)\s+([^\s]+)': r'let \1;',
        
        # Control flow
        r'IF\s+(.*?)\s+BEGIN': r'if (\1) {',
        r'END': r'}',
        r'WHILE\s+(.*?)\s+BEGIN': r'while (\1) {',
        r'RETURN\s+([^;]+)': r'return \1;',
        
        # Functions
        r'GETDATE\(\)': r'new Date()',
        r'CONVERT\s*\(\s*([^,]+),\s*([^)]+)\)': r'convertToType(\2, "\1")',
        r'ISNULL\s*\(([^,]+),\s*([^)]+)\)': r'(\1 !== null ? \1 : \2)',
        
        # Error handling
        r'RAISERROR\s*\(([^,)]+)(?:,\s*([^,)]+)(?:,\s*([^,)]+))?\)': 
            r'throw new Error(\1)',
        
        # Transaction management 
        r'BEGIN\s+TRAN(?:SACTION)?': r'const session = db.getMongo().startSession();\ntry {',
        r'COMMIT\s+TRAN(?:SACTION)?': r'session.commitTransaction();\n} finally {\n  session.endSession();\n}',
        r'ROLLBACK\s+TRAN(?:SACTION)?': r'session.abortTransaction();\n} finally {\n  session.endSession();\n}'
    }
    
    # Combine stored procedure mappings in one dictionary
    self.sp_mappings = {
        TargetDatabase.MSSQL: self.mssql_sp_mappings,
        TargetDatabase.ORACLE: self.oracle_sp_mappings,
        TargetDatabase.MONGODB: self.mongodb_sp_mappings
    }

def convert_stored_procedures(self, content, filename):
    """Convert Sybase stored procedures to target database format"""
    old_content = content
    changes = 0
    
    # For SQL files, we might have multiple procedures with GO or similar batch separators
    is_sql_file = filename.lower().endswith('.sql')
    
    if is_sql_file:
        # Split on GO batch separators if this is a SQL file
        batches = re.split(r'^\s*GO\s*$', content, flags=re.MULTILINE | re.IGNORECASE)
        converted_batches = []
        
        for batch in batches:
            if not batch.strip():
                converted_batches.append(batch)
                continue
                
            converted_batch = batch
            # Look for stored procedure definitions
            sp_pattern = re.compile(r'(CREATE\s+PROC(?:EDURE)?\s+\w+.*?END\s*;?)', re.IGNORECASE | re.MULTILINE | re.DOTALL)
            sp_matches = sp_pattern.finditer(batch)
            
            # Process each stored procedure in this batch
            replacements = []
            
            for match in sp_matches:
                original_proc = match.group(1)
                converted_proc = original_proc
                
                # Apply all mappings for the target database
                for sybase_pattern, target_replacement in self.sp_mappings[self.target_db].items():
                    converted_proc = re.sub(sybase_pattern, target_replacement, converted_proc, flags=re.IGNORECASE)
                
                # Database-specific modifications
                converted_proc = self._apply_db_specific_sp_changes(converted_proc)
                
                # Only add to replacements if something changed
                if original_proc != converted_proc:
                    replacements.append((match.start(), match.end(), 
                                        f"/* CONVERTED STORED PROCEDURE FROM SYBASE */\n{converted_proc}\n/* END CONVERTED PROCEDURE */"))
                    changes += 1
                    self.logger.info(f"{filename}: Converted stored procedure definition")
            
            # Apply replacements in reverse order
            for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
                converted_batch = converted_batch[:start] + replacement + converted_batch[end:]
            
            converted_batches.append(converted_batch)
        
        # Join batches with appropriate separator based on target database
        batch_separator = {
            TargetDatabase.MSSQL: "GO\n\n",
            TargetDatabase.ORACLE: "/\n\n",
            TargetDatabase.MONGODB: ";\n\n"  # For MongoDB we'll use semicolons
        }[self.target_db]
        
        content = batch_separator.join(converted_batches)
    else:
        # For C++ files, continue with the existing implementation
        # Look for stored procedure definitions and calls
        sp_pattern = re.compile(r'(CREATE\s+PROC(?:EDURE)?\s+[^;]+;)', re.IGNORECASE | re.MULTILINE | re.DOTALL)
        sp_calls = re.compile(r'EXEC(?:UTE)?\s+(\w+)(?:\s+([^;]*))?;', re.IGNORECASE)
        
        # Process stored procedure definitions
        sp_matches = sp_pattern.finditer(content)
        
        # Store procedure offsets to avoid conflicting replacements
        replacements = []
        
        for match in sp_matches:
            original_proc = match.group(1)
            converted_proc = original_proc
            
            # Apply all mappings for the target database
            for sybase_pattern, target_replacement in self.sp_mappings[self.target_db].items():
                converted_proc = re.sub(sybase_pattern, target_replacement, converted_proc, flags=re.IGNORECASE)
            
            # Apply database-specific modifications
            converted_proc = self._apply_db_specific_sp_changes(converted_proc)
            
            # Only add to replacements if something changed
            if original_proc != converted_proc:
                replacements.append((match.start(), match.end(), 
                                    f"/* CONVERTED STORED PROCEDURE FROM SYBASE */\n{converted_proc}\n/* END CONVERTED PROCEDURE */"))
                changes += 1
                self.logger.info(f"{filename}: Converted stored procedure definition")
        
        # Process stored procedure calls
        call_matches = sp_calls.finditer(content)
        
        for match in call_matches:
            original_call = match.group(0)
            proc_name = match.group(1)
            parameters = match.group(2) if match.group(2) else ""
            
            converted_call = self._convert_procedure_call(proc_name, parameters)
            
            if original_call != converted_call:
                replacements.append((match.start(), match.end(), 
                                    f"/* CONVERTED PROCEDURE CALL FROM SYBASE: {original_call} */\n{converted_call}\n/* END CONVERTED CALL */"))
                changes += 1
                self.logger.info(f"{filename}: Converted stored procedure call {proc_name}")
        
        # Apply replacements in reverse order to avoid offset issues
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            content = content[:start] + replacement + content[end:]
    
    if changes > 0:
        self.changes_made += changes
    
    return content

def _apply_db_specific_sp_changes(self, converted_proc):
    """Apply database-specific modifications to stored procedures"""
    if self.target_db == TargetDatabase.ORACLE:
        # Add BEGIN/END block if not present
        if "BEGIN" not in converted_proc.upper():
            if "AS" in converted_proc.upper():
                converted_proc = re.sub(r'(?i)(AS\s*)', r'\1\nBEGIN\n', converted_proc)
                converted_proc = converted_proc + "\nEND;"
            else:
                converted_proc = converted_proc + "\nAS\nBEGIN\nEND;"
    
    elif self.target_db == TargetDatabase.MONGODB:
        # Ensure MongoDB function ends with closing brace
        if not converted_proc.rstrip().endswith('}'):
            converted_proc = converted_proc + "\n}"
        # Add MongoDB function registration
        proc_name = re.search(r'function\s+(\w+)', converted_proc)
        if proc_name:
            converted_proc = converted_proc + f"\n\ndb.system.js.save({{ _id: '{proc_name.group(1)}', value: {proc_name.group(1)} }});"
    
    return converted_proc

def _convert_procedure_call(self, proc_name, parameters):
    """Convert a stored procedure call to the target database format"""
    if self.target_db == TargetDatabase.MSSQL:
        # MSSQL format is similar, just replace EXECUTE with EXEC
        return f"EXEC {proc_name} {parameters};"
    
    elif self.target_db == TargetDatabase.ORACLE:
        # Oracle uses BEGIN/END blocks for procedure calls
        return f"BEGIN\n  {proc_name}({parameters});\nEND;"
    
    elif self.target_db == TargetDatabase.MONGODB:
        # MongoDB calls JavaScript functions
        return f"db.eval('{proc_name}({parameters})')"

# Call this in main() before running the converter:
for db_type in [TargetDatabase.MSSQL, TargetDatabase.ORACLE, TargetDatabase.MONGODB]:
    target_db = db_type
    temp_converter = SybaseConverter(target_db, "temp.log")
    print(f"\nTesting patterns for {db_type.value}:")
    if not test_regex_patterns(temp_converter.function_mappings[target_db]):
        print(f"Found issue in {db_type.value} patterns")



if __name__ == "__main__":
    sys.exit(main())
