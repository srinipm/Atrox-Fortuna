#!/usr/bin/env python3
"""
Sybase Java JDBC Converter

This tool converts Java code using Sybase JDBC to equivalent code
for MS SQL Server, Oracle, PostgreSQL, or MongoDB.

Usage: python sybase_java_converter.py --input-dir /path/to/source --target [mssql|oracle|postgres|mongodb] [--backup-dir /path/to/backups]
"""

import re
import os
import sys
import argparse
import shutil
import logging
from enum import Enum
from datetime import datetime


class TargetDatabase(Enum):
    MSSQL = "mssql"
    ORACLE = "oracle"
    POSTGRES = "postgres"
    MONGODB = "mongodb"  # Added MongoDB support


class SybaseJavaConverter:
    def __init__(self, target_db, log_file):
        self.target_db = target_db
        self.logger = self._setup_logger(log_file)
        
        # Define replacements for imports
        self.import_replacements = {
            TargetDatabase.MSSQL: {
                r'import\s+com\.sybase\.jdbc\.\S+;': 'import com.microsoft.sqlserver.jdbc.SQLServerDriver;\nimport com.microsoft.sqlserver.jdbc.SQLServerDataSource;',
                r'import\s+com\.sybase\.jdbc4\.\S+;': 'import com.microsoft.sqlserver.jdbc.SQLServerDriver;\nimport com.microsoft.sqlserver.jdbc.SQLServerDataSource;',
                r'import\s+sybase\.jdbc\.\S+;': 'import com.microsoft.sqlserver.jdbc.SQLServerDriver;\nimport com.microsoft.sqlserver.jdbc.SQLServerDataSource;'
            },
            TargetDatabase.ORACLE: {
                r'import\s+com\.sybase\.jdbc\.\S+;': 'import oracle.jdbc.OracleDriver;\nimport oracle.jdbc.pool.OracleDataSource;',
                r'import\s+com\.sybase\.jdbc4\.\S+;': 'import oracle.jdbc.OracleDriver;\nimport oracle.jdbc.pool.OracleDataSource;',
                r'import\s+sybase\.jdbc\.\S+;': 'import oracle.jdbc.OracleDriver;\nimport oracle.jdbc.pool.OracleDataSource;'
            },
            TargetDatabase.POSTGRES: {
                r'import\s+com\.sybase\.jdbc\.\S+;': 'import org.postgresql.Driver;\nimport org.postgresql.ds.PGSimpleDataSource;',
                r'import\s+com\.sybase\.jdbc4\.\S+;': 'import org.postgresql.Driver;\nimport org.postgresql.ds.PGSimpleDataSource;',
                r'import\s+sybase\.jdbc\.\S+;': 'import org.postgresql.Driver;\nimport org.postgresql.ds.PGSimpleDataSource;'
            },
            TargetDatabase.MONGODB: {
                r'import\s+com\.sybase\.jdbc\.\S+;': 'import com.mongodb.client.MongoClients;\nimport com.mongodb.client.MongoClient;\nimport com.mongodb.client.MongoDatabase;\nimport com.mongodb.client.MongoCollection;\nimport org.bson.Document;\nimport com.mongodb.client.MongoCursor;',
                r'import\s+com\.sybase\.jdbc4\.\S+;': 'import com.mongodb.client.MongoClients;\nimport com.mongodb.client.MongoClient;\nimport com.mongodb.client.MongoDatabase;\nimport com.mongodb.client.MongoCollection;\nimport org.bson.Document;\nimport com.mongodb.client.MongoCursor;',
                r'import\s+sybase\.jdbc\.\S+;': 'import com.mongodb.client.MongoClients;\nimport com.mongodb.client.MongoClient;\nimport com.mongodb.client.MongoDatabase;\nimport com.mongodb.client.MongoCollection;\nimport org.bson.Document;\nimport com.mongodb.client.MongoCursor;'
            }
        }
        
        # Define driver class replacements
        self.driver_class_replacements = {
            TargetDatabase.MSSQL: {
                r'SybDriver': 'SQLServerDriver',
                r'com\.sybase\.jdbc\.SybDriver': 'com.microsoft.sqlserver.jdbc.SQLServerDriver',
                r'com\.sybase\.jdbc4\.SybDriver': 'com.microsoft.sqlserver.jdbc.SQLServerDriver'
            },
            TargetDatabase.ORACLE: {
                r'SybDriver': 'OracleDriver',
                r'com\.sybase\.jdbc\.SybDriver': 'oracle.jdbc.OracleDriver',
                r'com\.sybase\.jdbc4\.SybDriver': 'oracle.jdbc.OracleDriver'
            },
            TargetDatabase.POSTGRES: {
                r'SybDriver': 'Driver',
                r'com\.sybase\.jdbc\.SybDriver': 'org.postgresql.Driver',
                r'com\.sybase\.jdbc4\.SybDriver': 'org.postgresql.Driver'
            },
            TargetDatabase.MONGODB: {
                r'SybDriver': 'MongoClient',
                r'com\.sybase\.jdbc\.SybDriver': 'com.mongodb.client.MongoClients',
                r'com\.sybase\.jdbc4\.SybDriver': 'com.mongodb.client.MongoClients'
            }
        }
        
        # Define connection URL replacements
        self.connection_url_replacements = {
            TargetDatabase.MSSQL: {
                r'"jdbc:sybase:Tds:([^:]+)(?::(\d+))?(?:/([^"]+))?"': r'"jdbc:sqlserver://\1\2;\3"',
                r'"jdbc:sybase:Tds:([^:]+):(\d+)/([^"]+)"': r'"jdbc:sqlserver://\1:\2;databaseName=\3"'
            },
            TargetDatabase.ORACLE: {
                r'"jdbc:sybase:Tds:([^:]+)(?::(\d+))?(?:/([^"]+))?"': r'"jdbc:oracle:thin:@\1\2:\3"',
                r'"jdbc:sybase:Tds:([^:]+):(\d+)/([^"]+)"': r'"jdbc:oracle:thin:@\1:\2:\3"'
            },
            TargetDatabase.POSTGRES: {
                r'"jdbc:sybase:Tds:([^:]+)(?::(\d+))?(?:/([^"]+))?"': r'"jdbc:postgresql://\1\2/\3"',
                r'"jdbc:sybase:Tds:([^:]+):(\d+)/([^"]+)"': r'"jdbc:postgresql://\1:\2/\3"'
            },
            TargetDatabase.MONGODB: {
                r'"jdbc:sybase:Tds:([^:]+)(?::(\d+))?(?:/([^"]+))?"': r'"mongodb://\1\2/\3"',
                r'"jdbc:sybase:Tds:([^:]+):(\d+)/([^"]+)"': r'"mongodb://\1:\2/\3"'
            }
        }
        
        # Define connection property replacements
        self.connection_props_replacements = {
            TargetDatabase.MSSQL: {
                r'props\.put\("APPLICATIONNAME",\s*"([^"]+)"\)': r'props.put("applicationName", "\1")',
                r'props\.put\("USER",\s*"([^"]+)"\)': r'props.put("user", "\1")',
                r'props\.put\("PASSWORD",\s*"([^"]+)"\)': r'props.put("password", "\1")',
                r'props\.put\("LANGUAGE",\s*"([^"]+)"\)': r'// MSSQL doesn\'t use LANGUAGE property'
            },
            TargetDatabase.ORACLE: {
                r'props\.put\("APPLICATIONNAME",\s*"([^"]+)"\)': r'props.put("v$session.program", "\1")',
                r'props\.put\("USER",\s*"([^"]+)"\)': r'props.put("user", "\1")',
                r'props\.put\("PASSWORD",\s*"([^"]+)"\)': r'props.put("password", "\1")',
                r'props\.put\("LANGUAGE",\s*"([^"]+)"\)': r'// Oracle doesn\'t use LANGUAGE property'
            },
            TargetDatabase.POSTGRES: {
                r'props\.put\("APPLICATIONNAME",\s*"([^"]+)"\)': r'props.put("ApplicationName", "\1")',
                r'props\.put\("USER",\s*"([^"]+)"\)': r'props.put("user", "\1")',
                r'props\.put\("PASSWORD",\s*"([^"]+)"\)': r'props.put("password", "\1")',
                r'props\.put\("LANGUAGE",\s*"([^"]+)"\)': r'// PostgreSQL doesn\'t use LANGUAGE property'
            },
            TargetDatabase.MONGODB: {
                r'props\.put\("APPLICATIONNAME",\s*"([^"]+)"\)': r'// MongoDB connection options are specified in the connection string or MongoClientOptions',
                r'props\.put\("USER",\s*"([^"]+)"\)': r'// For MongoDB, set credentials in the connection string: mongodb://username:password@host',
                r'props\.put\("PASSWORD",\s*"([^"]+)"\)': r'// For MongoDB, set credentials in the connection string: mongodb://username:password@host',
                r'props\.put\("LANGUAGE",\s*"([^"]+)"\)': r'// MongoDB doesn\'t use LANGUAGE property'
            }
        }
        
        # Define stored procedure call replacements
        self.stored_procedure_replacements = {
            TargetDatabase.MSSQL: {
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({[^{}]+})"\s*\)': 
                    r'CallableStatement \1 = conn.prepareCall("{\2}")  // MSSQL uses same syntax',
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({call\s+[^{}]+})"\s*\)':
                    r'CallableStatement \1 = conn.prepareCall("\2")  // MSSQL uses same syntax'
            },
            TargetDatabase.ORACLE: {
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({[^{}]+})"\s*\)': 
                    r'CallableStatement \1 = conn.prepareCall("BEGIN \2; END;")  // Oracle uses BEGIN/END block',
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({call\s+([^{}]+)})"\s*\)':
                    r'CallableStatement \1 = conn.prepareCall("BEGIN \3; END;")  // Oracle uses BEGIN/END block'
            },
            TargetDatabase.POSTGRES: {
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({[^{}]+})"\s*\)': 
                    r'CallableStatement \1 = conn.prepareCall("SELECT \2")  // PostgreSQL uses SELECT for functions',
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({call\s+([^{}]+)})"\s*\)':
                    r'CallableStatement \1 = conn.prepareCall("SELECT \3")  // PostgreSQL uses SELECT for functions'
            },
            TargetDatabase.MONGODB: {
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({[^{}]+})"\s*\)': 
                    r'// MongoDB doesn\'t use stored procedures like SQL databases\n' + 
                    r'Document \1Params = new Document();\n' +
                    r'// Add parameters to document instead of using callable statement',
                r'CallableStatement\s+(\w+)\s*=\s*conn\.prepareCall\s*\(\s*"({call\s+([^{}]+)})"\s*\)':
                    r'// MongoDB doesn\'t use stored procedures like SQL databases\n' +
                    r'Document \1Params = new Document();\n' +
                    r'// For procedure "\3", add parameters to document instead'
            }
        }
        
        # Define result set meta data replacements
        self.result_set_replacements = {
            TargetDatabase.MSSQL: {},  # MSSQL and Sybase have similar result set handling
            TargetDatabase.ORACLE: {
                r'getString\("([^"]+)"\)': r'getString("\1".toUpperCase())'  # Oracle returns column names in uppercase
            },
            TargetDatabase.POSTGRES: {
                r'getString\("([^"]+)"\)': r'getString("\1".toLowerCase())'  # Postgres returns column names in lowercase
            },
            TargetDatabase.MONGODB: {
                r'ResultSet\s+(\w+)\s*=\s*(\w+)\.executeQuery\(\)': r'MongoCursor<Document> \1 = \2.find().iterator()',
                r'(\w+)\.getString\("([^"]+)"\)': r'\1.getString("\2") // In MongoDB, change to: \1.get("\2").toString()',
                r'(\w+)\.getInt\("([^"]+)"\)': r'\1.getInt("\2") // In MongoDB, change to: ((Integer)\1.get("\2")).intValue()',
                r'(\w+)\.getDouble\("([^"]+)"\)': r'\1.getDouble("\2") // In MongoDB, change to: ((Double)\1.get("\2")).doubleValue()',
                r'(\w+)\.next\(\)': r'\1.hasNext() ? \1.next() : null',
            }
        }
        
        # Define error handling replacements
        self.error_handling_replacements = {
            TargetDatabase.MSSQL: {
                r'catch\s*\(\s*com\.sybase\.jdbc\.SybSQLException\s+(\w+)\s*\)': 
                    r'catch (com.microsoft.sqlserver.jdbc.SQLServerException \1)'
            },
            TargetDatabase.ORACLE: {
                r'catch\s*\(\s*com\.sybase\.jdbc\.SybSQLException\s+(\w+)\s*\)': 
                    r'catch (java.sql.SQLException \1)'
            },
            TargetDatabase.POSTGRES: {
                r'catch\s*\(\s*com\.sybase\.jdbc\.SybSQLException\s+(\w+)\s*\)': 
                    r'catch (org.postgresql.util.PSQLException \1)'
            },
            TargetDatabase.MONGODB: {
                r'catch\s*\(\s*com\.sybase\.jdbc\.SybSQLException\s+(\w+)\s*\)': 
                    r'catch (com.mongodb.MongoException \1)'
            }
        }
        
        # MongoDB-specific SQL to MongoDB conversion patterns
        self.mongodb_specific_replacements = {
            # Convert SQL statements to MongoDB operations
            r'(String|StringBuilder)\s+(\w+)\s*=\s*(?:new\s+(?:String|StringBuilder)\s*\(\s*\))?\s*"SELECT\s+([^;]+)\s+FROM\s+([^;]+)(?:\s+WHERE\s+([^;]+))?"': 
                r'\1 \2 = "SELECT statement converted to MongoDB find()"\n' +
                r'// Original SQL: SELECT \3 FROM \4\5\n' +
                r'// MongoDB: collection.find(query)',
                
            # Convert JDBC statement execution to MongoDB find
            r'(\w+)\.executeQuery\(\s*"SELECT\s+([^;]+)\s+FROM\s+([^;]+)(?:\s+WHERE\s+([^;]+))?"[^)]*\)': 
                r'// Original SQL: \1.executeQuery("SELECT \2 FROM \3\4")\n' +
                r'// For MongoDB, use collection.find() instead:\n' +
                r'// MongoCollection<Document> collection = database.getCollection("\3");\n' +
                r'// Document query = new Document(); // Add query conditions from WHERE clause\n' +
                r'// MongoCursor<Document> cursor = collection.find(query).iterator();',
                
            # Convert JDBC updates to MongoDB update
            r'(\w+)\.executeUpdate\(\s*"UPDATE\s+([^;]+)\s+SET\s+([^;]+)(?:\s+WHERE\s+([^;]+))?"[^)]*\)': 
                r'// Original SQL: \1.executeUpdate("UPDATE \2 SET \3\4")\n' +
                r'// For MongoDB, use collection.updateMany() instead:\n' +
                r'// MongoCollection<Document> collection = database.getCollection("\2");\n' +
                r'// Document query = new Document(); // Add query conditions from WHERE clause\n' +
                r'// Document update = new Document("$set", new Document()); // Add fields from SET clause\n' +
                r'// collection.updateMany(query, update);',
                
            # Convert JDBC inserts to MongoDB insert
            r'(\w+)\.executeUpdate\(\s*"INSERT\s+INTO\s+([^;(]+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)"[^)]*\)': 
                r'// Original SQL: \1.executeUpdate("INSERT INTO \2(\3) VALUES(\4)")\n' +
                r'// For MongoDB, use collection.insertOne() instead:\n' +
                r'// MongoCollection<Document> collection = database.getCollection("\2");\n' +
                r'// Document doc = new Document(); // Add fields and values from columns and VALUES\n' +
                r'// collection.insertOne(doc);',
                
            # Convert JDBC deletes to MongoDB delete
            r'(\w+)\.executeUpdate\(\s*"DELETE\s+FROM\s+([^;]+)(?:\s+WHERE\s+([^;]+))?"[^)]*\)': 
                r'// Original SQL: \1.executeUpdate("DELETE FROM \2\3")\n' +
                r'// For MongoDB, use collection.deleteMany() instead:\n' +
                r'// MongoCollection<Document> collection = database.getCollection("\2");\n' +
                r'// Document query = new Document(); // Add query conditions from WHERE clause\n' +
                r'// collection.deleteMany(query);',
                
            # Convert JDBC Connection to MongoDB Client
            r'Connection\s+(\w+)\s*=\s*DriverManager\.getConnection\(([^)]+)\)': 
                r'// Original JDBC: Connection \1 = DriverManager.getConnection(\2)\n' +
                r'// For MongoDB:\n' +
                r'MongoClient \1Client = MongoClients.create(\2);\n' +
                r'MongoDatabase \1DB = \1Client.getDatabase("dbName"); // Specify database name',
                
            # Convert JDBC Statement to MongoDB Collection
            r'Statement\s+(\w+)\s*=\s*(\w+)\.createStatement\(\)': 
                r'// Original JDBC: Statement \1 = \2.createStatement()\n' +
                r'// For MongoDB, you work with collections instead:\n' +
                r'// MongoCollection<Document> \1Collection = \2DB.getCollection("collectionName");'
        }
        
        # Track changes made to files
        self.changes_made = 0

    def _setup_logger(self, log_file):
        """Set up file logger"""
        logger = logging.getLogger('sybase_java_converter')
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def convert_imports(self, content, filename):
        """Replace Sybase imports with target database imports"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.import_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase imports with {self.target_db.value} imports")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert_driver_class(self, content, filename):
        """Replace Sybase driver classes with target database driver classes"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.driver_class_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase driver references with {self.target_db.value} driver")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert_connection_urls(self, content, filename):
        """Replace Sybase connection URLs with target database connection URLs"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.connection_url_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase connection URLs with {self.target_db.value} URLs")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert_connection_properties(self, content, filename):
        """Replace Sybase connection properties with target database connection properties"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.connection_props_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase connection properties with {self.target_db.value} properties")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert_stored_procedures(self, content, filename):
        """Replace Sybase stored procedure calls with target database procedure calls"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.stored_procedure_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                # Process matches in reverse to avoid offset issues
                for match in reversed(matches):
                    # Get the original text
                    original_text = content[match.start():match.end()]
                    # Apply the replacement
                    replaced_text = re.sub(pattern, replacement, original_text)
                    # Add comments around the replacement
                    commented_replacement = f"/* CONVERTED FROM SYBASE: {original_text.strip()} */\n{replaced_text}\n/* END CONVERSION */"
                    # Replace in content
                    content = content[:match.start()] + commented_replacement + content[match.end():]
                    changes += 1
                    self.logger.info(f"{filename}: Converted stored procedure call")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert_result_sets(self, content, filename):
        """Replace Sybase result set handling with target database result set handling"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.result_set_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase result set handling with {self.target_db.value} equivalent")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert_error_handling(self, content, filename):
        """Replace Sybase error handling with target database error handling"""
        old_content = content
        changes = 0
        
        for pattern, replacement in self.error_handling_replacements[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase exception handling with {self.target_db.value} exceptions")
        
        if changes > 0:
            self.changes_made += changes
        
        return content
    
    def convert_specific_method_calls(self, content, filename):
        """Replace Sybase-specific method calls with target database equivalents"""
        old_content = content
        changes = 0
        
        # Map of Sybase-specific methods to target database methods
        method_mappings = {
            TargetDatabase.MSSQL: {
                r'setMaxRows\((\d+)\)': r'setMaxRows(\1)',  # Same in MSSQL
                r'setQueryTimeout\((\d+)\)': r'setQueryTimeout(\1)',  # Same in MSSQL
            },
            TargetDatabase.ORACLE: {
                r'setMaxRows\((\d+)\)': r'setMaxRows(\1)',  # Same in Oracle
                r'setQueryTimeout\((\d+)\)': r'setQueryTimeout(\1)',  # Same in Oracle
            },
            TargetDatabase.POSTGRES: {
                r'setMaxRows\((\d+)\)': r'setFetchSize(\1)',  # PostgreSQL uses setFetchSize instead
                r'setQueryTimeout\((\d+)\)': r'setQueryTimeout(\1)',  # Same in PostgreSQL
            },
            TargetDatabase.MONGODB: {
                r'setMaxRows\((\d+)\)': r'// MongoDB equivalent: collection.find().limit(\1)',
                r'setQueryTimeout\((\d+)\)': r'// MongoDB uses maxTimeMS: collection.find().maxTimeMS(\1 * 1000)',
            }
        }
        
        for pattern, replacement in method_mappings[self.target_db].items():
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                changes += len(matches)
                self.logger.info(f"{filename}: Replaced {len(matches)} Sybase-specific methods with {self.target_db.value} methods")
        
        if changes > 0:
            self.changes_made += changes
        
        return content
    
    def convert_mongodb_specific(self, content, filename):
        """Apply MongoDB-specific replacements for SQL to MongoDB conversion"""
        # Only apply these if target is MongoDB
        if self.target_db != TargetDatabase.MONGODB:
            return content
            
        old_content = content
        changes = 0
        
        for pattern, replacement in self.mongodb_specific_replacements.items():
            matches = list(re.finditer(pattern, content))
            if matches:
                # Process matches in reverse to avoid offset issues
                for match in reversed(matches):
                    # Get the original text
                    original_text = content[match.start():match.end()]
                    # Apply the replacement
                    replaced_text = re.sub(pattern, replacement, original_text)
                    # Add comments around the replacement
                    commented_replacement = f"/* CONVERTED FROM SQL TO MONGODB: {original_text.strip()} */\n{replaced_text}\n/* END CONVERSION */"
                    # Replace in content
                    content = content[:match.start()] + commented_replacement + content[match.end():]
                    changes += 1
                    self.logger.info(f"{filename}: Converted SQL operation to MongoDB equivalent")
        
        if changes > 0:
            self.changes_made += changes
        
        return content

    def convert(self, input_content, filename):
        """Convert Sybase Java JDBC code to target database code"""
        result = input_content
        
        # Apply transformations in sequence
        result = self.convert_imports(result, filename)
        result = self.convert_driver_class(result, filename)
        result = self.convert_connection_urls(result, filename)
        result = self.convert_connection_properties(result, filename)
        result = self.convert_stored_procedures(result, filename)
        result = self.convert_result_sets(result, filename)
        result = self.convert_error_handling(result, filename)
        result = self.convert_specific_method_calls(result, filename)
        
        # Apply MongoDB-specific transformations if target is MongoDB
        if self.target_db == TargetDatabase.MONGODB:
            result = self.convert_mongodb_specific(result, filename)
        
        # Add header comment for converted files
        target_name = {
            TargetDatabase.MSSQL: "Microsoft SQL Server",
            TargetDatabase.ORACLE: "Oracle",
            TargetDatabase.POSTGRES: "PostgreSQL",
            TargetDatabase.MONGODB: "MongoDB"
        }[self.target_db]
        
        if self.changes_made > 0:
            header = f"""/**
 * @file {os.path.basename(filename)}
 * 
 * JDBC code converted from Sybase to {target_name} by Sybase Java JDBC Converter
 * Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

"""
            # Only add the header if it doesn't start with a comment already
            if not result.lstrip().startswith("/*") and not result.lstrip().startswith("//"):
                result = header + result
        
        return result


def find_java_files(directory):
    """Find all Java source files in a directory and its subdirectories"""
    java_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files


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
    parser = argparse.ArgumentParser(description='Convert Sybase Java JDBC code to other database APIs')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing Java files')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for converted files')
    parser.add_argument('--backup-dir', '-b', default='java_backups', help='Directory to store file backups (default: ./java_backups)')
    parser.add_argument('--target', '-t', required=True, choices=['mssql', 'oracle', 'postgres', 'mongodb'],
                        help='Target database system')
    
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
    log_file = f"java_conversion_{args.target}_{timestamp}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Find all Java source files
    java_files = find_java_files(args.input_dir)
    if not java_files:
        print(f"No Java files found in '{args.input_dir}'")
        return 1
    
    print(f"Found {len(java_files)} Java files to process")
    
    # Initialize converter
    target_db = TargetDatabase(args.target)
    converter = SybaseJavaConverter(target_db, log_file)
    
    # Process each file
    files_processed = 0
    files_changed = 0
    backup_dir = os.path.abspath(args.backup_dir)
    
    for file_path in java_files:
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
            
            # Process the file
            output_content = converter.convert(input_content, file_path)
            
            files_processed += 1
            
            # Check if file was actually changed
            if converter.changes_made > initial_changes:
                # Create backup before modifying
                backup_path = backup_file(file_path, backup_dir)
                converter.logger.info(f"Created backup at: {backup_path}")
                
                files_changed += 1
                converter.logger.info(f"Successfully converted: {file_path}")
                print(f"  - Modified file with {converter.changes_made - initial_changes} changes")
                
                # Write to output directory, preserving directory structure
                rel_path = os.path.relpath(file_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Write converted content
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)
            else:
                print(f"  - No changes needed")
                converter.logger.info(f"No changes made to: {file_path}")
                
                # Copy file to output directory anyway
                rel_path = os.path.relpath(file_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(file_path, output_path)
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            converter.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
    
    # Summary
    print(f"\nConversion complete:")
    print(f"  - Processed {files_processed} Java files")
    print(f"  - Modified {files_changed} files")
    print(f"  - Made {converter.changes_made} total changes")
    print(f"  - Converted files saved to {os.path.abspath(args.output_dir)}")
    print(f"  - Backups stored in {backup_dir}")
    print(f"  - Detailed log saved to {log_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
