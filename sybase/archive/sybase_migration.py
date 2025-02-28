#!/usr/bin/env python3
"""
Sybase Database Migration Tool

A comprehensive tool for migrating from Sybase to other database systems.
Features:
- Code Analysis: Scans C++ source files for Sybase database calls
- Multi-Target Support: Migrate to Microsoft SQL Server, Oracle, or MongoDB
- Schema Translation: Converts Sybase DDL to target database syntax
- Data Migration: Generates scripts for data extraction and loading
- Reporting: Creates detailed migration reports and reference guides

Usage:
    python sybase_migrator.py /path/to/source/code --target mssql --extract-ddl --data-migration

Author: Claude AI
License: MIT
Version: 1.0.0
"""

import os
import re
import sys
import csv
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('migration.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Migrate Sybase database calls in C++ code to another database system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan and report only
  python sybase_migrator.py /path/to/source/code -o report.txt
  
  # Scan and modify code to use SQL Server
  python sybase_migrator.py /path/to/source/code --target mssql
  
  # Comprehensive migration including schema and data
  python sybase_migrator.py /path/to/source/code --target oracle --recursive --extract-ddl --data-migration
  
  # Dry run with MongoDB as target
  python sybase_migrator.py /path/to/source/code --target mongodb --dry-run
"""
    )
    parser.add_argument(
        'source_dir',
        type=str,
        help='Directory containing C++ source files to scan and modify'
    )
    parser.add_argument(
        '-t', '--target',
        type=str,
        choices=['mssql', 'oracle', 'mongodb'],
        default='mssql',
        help='Target database system for migration (default: mssql)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='migration_report.txt',
        help='Output report file name (default: migration_report.txt)'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Scan directories recursively'
    )
    parser.add_argument(
        '-e', '--extensions',
        type=str,
        default='.cpp,.cxx,.cc,.h,.hpp',
        help='Comma-separated list of file extensions to scan (default: .cpp,.cxx,.cc,.h,.hpp)'
    )
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Perform dry run without modifying files (only generate report)'
    )
    parser.add_argument(
        '-b', '--backup-dir',
        type=str,
        default='sybase_backup',
        help='Directory to store original file backups (default: sybase_backup)'
    )
    parser.add_argument(
        '--extract-ddl',
        action='store_true',
        help='Extract DDL statements from code and generate target database schema scripts'
    )
    parser.add_argument(
        '--data-migration',
        action='store_true',
        help='Generate data migration scripts for the target database'
    )
    parser.add_argument(
        '--db-connection',
        type=str,
        help='Database connection string for extracting schema information (if needed)'
    )
    parser.add_argument(
        '--schema-output-dir',
        type=str,
        default='schema_migration',
        help='Directory to output schema migration scripts (default: schema_migration)'
    )
    parser.add_argument(
        '--data-output-dir',
        type=str,
        default='data_migration',
        help='Directory to output data migration scripts (default: data_migration)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Sybase Migration Tool v1.0.0'
    )
    return parser.parse_args()


class MigrationHandler:
    """Base class for database migration handlers."""
    
    def __init__(self):
        self.patterns = []
        self.equivalents_dict = {}
        self.migration_notes = []
        self.ddl_patterns = []
    
    def get_patterns(self):
        """Return regex patterns with replacement functions."""
        return self.patterns
    
    def get_equivalents(self):
        """Return dictionary of equivalents."""
        return self.equivalents_dict
    
    def get_migration_notes(self):
        """Return list of migration notes."""
        return self.migration_notes
    
    def get_target_name(self):
        """Return the name of the target database system."""
        raise NotImplementedError("Subclasses must implement get_target_name()")
    
    def get_ddl_patterns(self):
        """Return regex patterns to identify DDL statements."""
        return self.ddl_patterns
    
    def generate_schema_script(self, extracted_ddl, output_dir):
        """Generate schema creation script for the target database."""
        raise NotImplementedError("Subclasses must implement generate_schema_script()")
    
    def generate_data_migration_script(self, table_structures, output_dir):
        """Generate data migration script for the target database."""
        raise NotImplementedError("Subclasses must implement generate_data_migration_script()")

    def translate_data_type(self, sybase_type):
        """Translate Sybase data type to target database equivalent."""
        raise NotImplementedError("Subclasses must implement translate_data_type()")


class MSSQLMigrationHandler(MigrationHandler):
    """Handler for migration to Microsoft SQL Server."""
    
    def __init__(self):
        super().__init__()
        self._init_patterns()
        self._init_equivalents()
        self._init_migration_notes()
        self._init_ddl_patterns()
    
    def get_target_name(self):
        return "Microsoft SQL Server"
    
    def _init_patterns(self):
        # Define helper functions to generate replacements
        def replace_cs_function(match):
            func_name = match.group(0).strip()
            return f"/* MSSQL Migration: {func_name} -> Use ODBC or SQL Server Native Client */\n"
        
        def replace_ct_function(match):
            func_name = match.group(0).strip()
            return f"/* MSSQL Migration: {func_name} -> Use SQLServer Native Client functions */\n"
        
        def replace_db_function(match):
            func_name = match.group(1).strip()
            return f"/* MSSQL Migration: {func_name} -> Use SQL Server equivalent (SQLConnect, SQLExecDirect, etc.) */\n"
        
        def replace_bcp_function(match):
            func_name = match.group(1).strip()
            return f"/* MSSQL Migration: {func_name} -> Use SQLBulkOperations or bcp_* SQL Server functions */\n"
        
        def replace_srv_function(match):
            func_name = match.group(1).strip()
            return f"/* MSSQL Migration: {func_name} -> Consider SQL Server Extended Stored Procedure API */\n"
        
        def replace_class(match):
            class_name = match.group(1).strip()
            return f"/* MSSQL Migration: {class_name} -> Use SQL Server Native Client or ODBC classes */\n"
        
        def replace_connect(match):
            conn_str = match.group(0).strip()
            return f"/* MSSQL Migration: {conn_str} -> Use SQL Server connection string */\n"
        
        def replace_isql(match):
            isql_cmd = match.group(0).strip()
            return f"/* MSSQL Migration: {isql_cmd} -> Use sqlcmd */\n"
        
        def replace_sybase_header(match):
            header = match.group(0).strip()
            if 'sybase.h' in header.lower():
                return f"/* MSSQL Migration: {header} -> Include SQL Server headers (sqlncli.h, sql.h, etc.) */\n"
            elif 'sybdb.h' in header.lower():
                return f"/* MSSQL Migration: {header} -> Include SQL Server headers (sqlncli.h, sql.h) */\n"
            else:
                return f"/* MSSQL Migration: {header} -> Include ODBC or SQL Server Native Client headers */\n"
        
        def replace_sql_statement(match):
            sql_keyword = match.group(1).strip()
            return f"/* MSSQL Migration: May need syntax adjustments for {sql_keyword} statement */\n"
        
        def replace_exec_sql(match):
            exec_stmt = match.group(0).strip()
            return f"/* MSSQL Migration: {exec_stmt} -> Use ODBC SQLExecDirect or SQL Server equivalent */\n"
        
        def replace_prepare_execute(match):
            prepare_stmt = match.group(0).strip()
            return f"/* MSSQL Migration: {prepare_stmt} -> Use SQLPrepare/SQLExecute or SqlCommand with parameters */\n"
        
        def replace_stored_proc(match):
            sp_stmt = match.group(0).strip()
            return f"/* MSSQL Migration: {sp_stmt} -> Use SQL Server stored procedure syntax (may need adjustments) */\n"
        
        # Sybase API calls and their replacements
        self.patterns = [
            (r'CS_\w+\s*\(', replace_cs_function),
            (r'ct_\w+\s*\(', replace_ct_function),
            (r'blk_\w+\s*\(', replace_ct_function),
            (r'(db(?:open|close|cmd|curs|data|info|login|options|props|results|sqlexec|stringinit|text|wrtext))\s*\(', replace_db_function),
            (r'(bcp_\w+)\s*\(', replace_bcp_function),
            (r'(srv_\w+)\s*\(', replace_srv_function),
            (r'(CSybase|CTlib|CDBLibrary|CSymbol)\s*[:\(]', replace_class),
            (r'connect\s+to\s+[\'"]?\w+[\'"]?', replace_connect),
            (r'isql\s+(-\w+\s+)*', replace_isql),
            (r'(?:ASE|openclient|sybase\.h|sybdb\.h|sybfront\.h)', replace_sybase_header),
            
            # SQL statements
            (r'(SELECT|INSERT|UPDATE|DELETE|EXECUTE|EXEC|CREATE|ALTER|DROP|TRUNCATE|BEGIN|COMMIT|ROLLBACK)\s+', replace_sql_statement),
            (r'EXEC\s+SQL', replace_exec_sql),
            (r'prepare\s+.+\s+from', replace_prepare_execute),
            (r'execute\s+.+\s+using', replace_prepare_execute),
            (r'exec_sql\(', replace_exec_sql),
            (r'execute\s+immediate', replace_exec_sql),
            (r'exec\s+sp_\w+', replace_stored_proc),
            (r'execute\s+sp_\w+', replace_stored_proc),
        ]
        
        # Compile patterns
        self.patterns = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in self.patterns]
    
    def _init_ddl_patterns(self):
        """Initialize patterns to identify DDL statements in code."""
        self.ddl_patterns = [
            # Table creation
            (re.compile(r'CREATE\s+TABLE\s+(\w+)[\s\(]', re.IGNORECASE), 'table_create'),
            # Table alteration
            (re.compile(r'ALTER\s+TABLE\s+(\w+)', re.IGNORECASE), 'table_alter'),
            # Table drop
            (re.compile(r'DROP\s+TABLE\s+(\w+)', re.IGNORECASE), 'table_drop'),
            # Index creation
            (re.compile(r'CREATE\s+(UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)', re.IGNORECASE), 'index_create'),
            # Stored procedure creation
            (re.compile(r'CREATE\s+PROC(?:EDURE)?\s+(\w+)', re.IGNORECASE), 'proc_create'),
            # View creation
            (re.compile(r'CREATE\s+VIEW\s+(\w+)', re.IGNORECASE), 'view_create'),
            # Column definition pattern (for data type extraction)
            (re.compile(r'(\w+)\s+((?:varchar|char|nvarchar|nchar|int|bigint|smallint|tinyint|numeric|decimal|money|float|real|datetime|smalldatetime|date|time|bit|text|image|binary|varbinary)\s*(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?)(\s+(?:NULL|NOT\s+NULL))?', re.IGNORECASE), 'column_def')
        ]
        
    def _init_equivalents(self):
        self.equivalents_dict = {
            # API/Function mappings
            "CS_*": ["SQLConnect", "SQLDriverConnect", "SQL Server Native Client"],
            "ct_*": ["SQL Server Native Client functions", "ODBC API functions"],
            "db*": ["SQLConnect", "SQLExecDirect", "SQL Native Client"],
            "bcp_*": ["SQLBulkOperations", "SQL Server bcp utility"],
            
            # Connection syntax
            "connect to": ["SQLDriverConnect", "SqlConnection.Open()"],
            "isql": ["sqlcmd"],
            
            # Headers
            "sybase.h": ["sql.h", "sqlext.h", "sqlncli.h"],
            "sybdb.h": ["sql.h", "sqlext.h"],
            "sybfront.h": ["sql.h", "sqlncli.h"],
            
            # SQL Syntax differences
            "SELECT TOP n": ["SELECT TOP n", "No change needed"],
            "SELECT FIRST n": ["SELECT TOP n"],
            "@@identity": ["SCOPE_IDENTITY()"],
            "getdate()": ["GETDATE()", "No change needed"],
            "convert()": ["CONVERT()", "Argument order might differ"],
            "sp_*": ["Similar stored procedures exist but may have different arguments"],
            
            # Transaction handling
            "BEGIN TRANSACTION": ["BEGIN TRANSACTION", "No change needed"],
            "COMMIT TRANSACTION": ["COMMIT TRANSACTION", "No change needed"],
            "ROLLBACK TRANSACTION": ["ROLLBACK TRANSACTION", "No change needed"],
            
            # Data types
            "datetime": ["datetime", "datetime2", "Note: datetime2 is recommended for new dev"],
            "smalldatetime": ["smalldatetime", "No change needed"],
            "money": ["money", "No change needed"],
            "text": ["text (deprecated)", "varchar(max)", "nvarchar(max)"],
            "image": ["image (deprecated)", "varbinary(max)"],
            
            # System functions
            "@@version": ["@@VERSION"],
            "@@servername": ["@@SERVERNAME"],
            "@@spid": ["@@SPID", "No change needed"],
        }
    
    def _init_migration_notes(self):
        self.migration_notes = [
            "Microsoft SQL Server has different transaction semantics than Sybase.",
            "Consider using ORM frameworks to abstract database access for new code.",
            "Parameter syntax (@param vs :param) differs between Sybase and SQL Server.",
            "Some Sybase system stored procedures have no direct SQL Server equivalents.",
            "SQL Server has different locking and isolation level behavior.",
            "Date/time functions and formats may have subtle differences.",
            "SQL Server uses square brackets [ ] for delimited identifiers instead of double quotes.",
            "IDENTITY columns work differently from Sybase IDENTITY columns."
        ]
    
    def translate_data_type(self, sybase_type):
        """Translate Sybase data type to SQL Server equivalent."""
        sybase_type = sybase_type.lower().strip()
        
        # Define mappings
        type_map = {
            'varchar': 'varchar',  # Same type
            'char': 'char',  # Same type
            'nvarchar': 'nvarchar',  # Same type
            'nchar': 'nchar',  # Same type
            'int': 'int',  # Same type
            'bigint': 'bigint',  # Same type
            'smallint': 'smallint',  # Same type
            'tinyint': 'tinyint',  # Same type
            'numeric': 'numeric',  # Same type
            'decimal': 'decimal',  # Same type
            'money': 'money',  # Same type
            'smallmoney': 'smallmoney',  # Same type
            'float': 'float',  # Same type
            'real': 'real',  # Same type
            'datetime': 'datetime2',  # Recommended over datetime
            'smalldatetime': 'smalldatetime',  # Same type
            'date': 'date',  # Same type
            'time': 'time',  # Same type
            'bit': 'bit',  # Same type
            'text': 'varchar(max)',  # text is deprecated in SQL Server
            'unitext': 'nvarchar(max)',  # unitext is specific to Sybase
            'image': 'varbinary(max)',  # image is deprecated in SQL Server
            'binary': 'binary',  # Same type
            'varbinary': 'varbinary',  # Same type
            'timestamp': 'rowversion',  # timestamp in SQL Server is now rowversion
        }
        
        # Extract the base type and size/precision if present
        base_type_match = re.match(r'(\w+)(?:\(([^)]+)\))?', sybase_type)
        if not base_type_match:
            return sybase_type  # Return as is if pattern doesn't match
        
        base_type = base_type_match.group(1)
        size_precision = base_type_match.group(2) if base_type_match.group(2) else None
        
        # Get the SQL Server type
        sql_server_type = type_map.get(base_type, base_type)  # Default to same type if not in map
        
        # Add size/precision if it exists
        if size_precision and sql_server_type not in ['text', 'image', 'timestamp']:
            return f"{sql_server_type}({size_precision})"
        
        return sql_server_type
    
    def generate_schema_script(self, extracted_ddl, output_dir):
        """Generate SQL Server schema creation script from extracted DDL."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a script for each object type
        tables_script_path = os.path.join(output_dir, "01_tables_mssql.sql")
        indexes_script_path = os.path.join(output_dir, "02_indexes_mssql.sql")
        procs_script_path = os.path.join(output_dir, "03_procedures_mssql.sql")
        views_script_path = os.path.join(output_dir, "04_views_mssql.sql")
        
        with open(tables_script_path, 'w', encoding='utf-8') as tables_file, \
             open(indexes_script_path, 'w', encoding='utf-8') as indexes_file, \
             open(procs_script_path, 'w', encoding='utf-8') as procs_file, \
             open(views_script_path, 'w', encoding='utf-8') as views_file:
            
            # Write header
            header = f"""-- SQL Server Schema Migration Script
-- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
-- Target: Microsoft SQL Server

"""
            tables_file.write(header)
            indexes_file.write(header)
            procs_file.write(header)
            views_file.write(header)
            
            # Process each DDL statement
            for ddl_type, objects in extracted_ddl.items():
                if ddl_type == 'table_create':
                    for name, ddl in objects.items():
                        mssql_ddl = self._convert_table_ddl(name, ddl)
                        tables_file.write(f"-- Original Sybase table: {name}\n")
                        tables_file.write(f"{mssql_ddl}\nGO\n\n")
                
                elif ddl_type == 'index_create':
                    for name, ddl in objects.items():
                        mssql_ddl = self._convert_index_ddl(ddl)
                        indexes_file.write(f"-- Original Sybase index: {name}\n")
                        indexes_file.write(f"{mssql_ddl}\nGO\n\n")
                
                elif ddl_type == 'proc_create':
                    for name, ddl in objects.items():
                        mssql_ddl = self._convert_proc_ddl(name, ddl)
                        procs_file.write(f"-- Original Sybase procedure: {name}\n")
                        procs_file.write(f"{mssql_ddl}\nGO\n\n")
                
                elif ddl_type == 'view_create':
                    for name, ddl in objects.items():
                        mssql_ddl = self._convert_view_ddl(name, ddl)
                        views_file.write(f"-- Original Sybase view: {name}\n")
                        views_file.write(f"{mssql_ddl}\nGO\n\n")
        
        # Create a master script that references all the other scripts
        master_script_path = os.path.join(output_dir, "00_master_mssql.sql")
        with open(master_script_path, 'w', encoding='utf-8') as master_file:
            master_file.write(f"""-- SQL Server Schema Migration Master Script
-- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
-- Target: Microsoft SQL Server

PRINT 'Starting SQL Server schema migration...'

PRINT 'Creating tables...'
:r ./01_tables_mssql.sql

PRINT 'Creating indexes...'
:r ./02_indexes_mssql.sql

PRINT 'Creating stored procedures...'
:r ./03_procedures_mssql.sql

PRINT 'Creating views...'
:r ./04_views_mssql.sql

PRINT 'Schema migration completed.'
""")
        
        return {
            "master": master_script_path,
            "tables": tables_script_path,
            "indexes": indexes_script_path,
            "procedures": procs_script_path,
            "views": views_script_path
        }
    
    def _convert_table_ddl(self, table_name, ddl):
        """Convert Sybase table DDL to SQL Server format."""
        # Replace Sybase-specific syntax with SQL Server syntax
        result = ddl
        
        # Convert data types
        column_def_pattern = re.compile(r'(\w+)\s+((?:varchar|char|nvarchar|nchar|int|bigint|smallint|tinyint|numeric|decimal|money|float|real|datetime|smalldatetime|date|time|bit|text|image|binary|varbinary)\s*(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?)', re.IGNORECASE)
        
        for match in column_def_pattern.finditer(ddl):
            column_name = match.group(1)
            sybase_type = match.group(2)
            mssql_type = self.translate_data_type(sybase_type)
            
            # Replace the data type in the result
            if sybase_type != mssql_type:
                result = result.replace(f"{column_name} {sybase_type}", f"{column_name} {mssql_type}")
        
        # Replace Sybase-specific identity syntax if present
        identity_pattern = re.compile(r'IDENTITY\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', re.IGNORECASE)
        result = identity_pattern.sub(r'IDENTITY(\1, \2)', result)
        
        # Replace double quotes with square brackets for identifiers
        # This is a simplistic approach; a proper parser would be better
        result = re.sub(r'"(\w+)"', r'[\1]', result)
        
        return result
    
    def _convert_index_ddl(self, ddl):
        """Convert Sybase index DDL to SQL Server format."""
        # Most index syntax is compatible, just change quotes to brackets
        result = re.sub(r'"(\w+)"', r'[\1]', ddl)
        return result
    
    def _convert_proc_ddl(self, proc_name, ddl):
        """Convert Sybase stored procedure DDL to SQL Server format."""
        # Replace Sybase-specific stored procedure syntax
        result = ddl
        
        # Replace Sybase's AS keyword with SQL Server's BEGIN...END block if not present
        if not re.search(r'\bBEGIN\b', result, re.IGNORECASE):
            result = re.sub(r'(\bAS\b)', r'\1\nBEGIN', result, flags=re.IGNORECASE)
            result += "\nEND"
        
        # Replace Sybase's return values with SQL Server style
        result = re.sub(r'RETURN\s+(\d+)', r'RETURN \1', result)
        
        # Replace double quotes with square brackets for identifiers
        result = re.sub(r'"(\w+)"', r'[\1]', result)
        
        return result
    
    def _convert_view_ddl(self, view_name, ddl):
        """Convert Sybase view DDL to SQL Server format."""
        # Most view syntax is compatible, just change quotes to brackets
        result = re.sub(r'"(\w+)"', r'[\1]', ddl)
        return result
    
    def generate_data_migration_script(self, table_structures, output_dir):
        """Generate SQL Server data migration scripts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a script for each table
        scripts = {}
        for table_name, columns in table_structures.items():
            script_path = os.path.join(output_dir, f"{table_name}_data_mssql.sql")
            scripts[table_name] = script_path
            
            with open(script_path, 'w', encoding='utf-8') as script_file:
                script_file.write(f"""-- SQL Server Data Migration Script for {table_name}
-- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
-- Target: Microsoft SQL Server

-- Ensure the destination table exists
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}')
BEGIN
    RAISERROR('Error: Destination table {table_name} does not exist.', 16, 1);
    RETURN;
END

-- BCP command template to export data from Sybase
/*
    Use the following command to export data from Sybase:
    
    bcp "{table_name}" out "{table_name}.dat" -c -t"," -r"\\n" -S<sybase_server> -U<username> -P<password>
*/

-- Bulk insert command to import data into SQL Server
/*
    Use the following command to import data into SQL Server:
    
    BULK INSERT {table_name}
    FROM '{table_name}.dat'
    WITH (
        FIELDTERMINATOR = ',',
        ROWTERMINATOR = '\\n',
        KEEPNULLS,
        TABLOCK
    );
*/

-- Alternative SQL approach using OPENROWSET
/*
    INSERT INTO {table_name} ({', '.join(columns)})
    SELECT {', '.join(columns)}
    FROM OPENROWSET(BULK '{table_name}.dat', 
                   FORMATFILE = '{table_name}_format.fmt') AS src;
*/

-- Sample format file content for {table_name}_format.fmt:
/*
{len(columns) + 1}
1 SQLCHAR 0 0 "," 1 dummy SQL_Latin1_General_CP1_CI_AS
""")

                # Write column format info
                for i, column in enumerate(columns, 2):
                    script_file.write(f"{i} SQLCHAR 0 0 \"\\n\" {i - 1} {column} SQL_Latin1_General_CP1_CI_AS\n")
                
                script_file.write("*/\n\n")
                
                # Create data validation section
                script_file.write(f"""-- Data validation after import
SELECT COUNT(*) AS RecordCount FROM {table_name};
-- Verify expected record count matches

-- Sample data verification query
SELECT TOP 10 * FROM {table_name};
-- Visually inspect sample data
""")
        
        # Create a master script
        master_script_path = os.path.join(output_dir, "00_master_data_migration_mssql.sql")
        with open(master_script_path, 'w', encoding='utf-8') as master_file:
            master_file.write(f"""-- SQL Server Data Migration Master Script
-- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
-- Target: Microsoft SQL Server

PRINT 'Starting SQL Server data migration...'

-- Disable constraints and triggers for faster loading
PRINT 'Disabling constraints and triggers...'
EXEC sp_MSforeachtable 'ALTER TABLE ? NOCHECK CONSTRAINT ALL'
EXEC sp_MSforeachtable 'ALTER TABLE ? DISABLE TRIGGER ALL'

-- Import data for each table
""")
            
            for table_name in table_structures.keys():
                master_file.write(f"PRINT 'Importing data for table {table_name}...'\n")
                master_file.write(f":r ./{table_name}_data_mssql.sql\n\n")
            
            master_file.write("""-- Re-enable constraints and triggers
PRINT 'Re-enabling constraints and triggers...'
EXEC sp_MSforeachtable 'ALTER TABLE ? CHECK CONSTRAINT ALL'
EXEC sp_MSforeachtable 'ALTER TABLE ? ENABLE TRIGGER ALL'

-- Validate referential integrity
PRINT 'Validating referential integrity...'
""")