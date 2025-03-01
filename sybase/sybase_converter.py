#!/usr/bin/env python3
"""
Sybase Database Converter

This script converts Sybase stored procedures to other database formats 
(MS SQL, Oracle, or Java).
It reads files from an input directory, applies conversion rules,
and writes the converted procedures to an output directory.
Changes are logged to a timestamped log file, and comments are added
to mark changes from the original code.

Usage:
    python sybase_converter.py --input <input_dir> --output <output_dir> --target <target_db>
"""

import os
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path


class SybaseConverter:
    """Converts Sybase stored procedures to other database formats."""

    def __init__(self, logger, target_db='mssql'):
        self.logger = logger
        self.target_db = target_db.lower()
        
        # Select appropriate conversion rules based on target database
        if self.target_db == 'mssql':
            self.conversion_rules = self._get_mssql_rules()
        elif self.target_db == 'oracle':
            self.conversion_rules = self._get_oracle_rules()
        elif self.target_db == 'java':
            self.conversion_rules = self._get_java_rules()
        elif self.target_db == 'javascript':
            self.conversion_rules = self._get_javascript_rules()
        else:
            self.logger.error(f"Unsupported target database: {target_db}")
            self.conversion_rules = {}
            
        self.logger.info(f"Initialized converter for target: {self.target_db}")

    def _get_mssql_rules(self):
        """Get conversion rules for MS SQL target."""
        return {
            # Data type conversions
            r'\bINTEGER\b': ('INT', 'Changed INTEGER to INT'),
            r'\bSMALLINT\b': ('SMALLINT', 'Kept SMALLINT (compatible)'),
            r'\bTINYINT\b': ('TINYINT', 'Kept TINYINT (compatible)'),
            r'\bDECIMAL\b': ('DECIMAL', 'Kept DECIMAL (compatible)'),
            r'\bNUMERIC\b': ('NUMERIC', 'Kept NUMERIC (compatible)'),
            r'\bMONEY\b': ('MONEY', 'Kept MONEY (compatible)'),
            r'\bSMALLMONEY\b': ('SMALLMONEY', 'Kept SMALLMONEY (compatible)'),
            r'\bFLOAT\b': ('FLOAT', 'Kept FLOAT (compatible)'),
            r'\bREAL\b': ('REAL', 'Kept REAL (compatible)'),
            r'\bDATETIME\b': ('DATETIME', 'Kept DATETIME (compatible)'),
            r'\bSMALLDATETIME\b': ('SMALLDATETIME', 'Kept SMALLDATETIME (compatible)'),
            r'\bDATE\b': ('DATE', 'Kept DATE (compatible)'),
            r'\bTIME\b': ('TIME', 'Kept TIME (compatible)'),
            r'\bCHAR\b': ('CHAR', 'Kept CHAR (compatible)'),
            r'\bVARCHAR\b': ('VARCHAR', 'Kept VARCHAR (compatible)'),
            r'\bTEXT\b': ('VARCHAR(MAX)', 'Changed TEXT to VARCHAR(MAX)'),
            r'\bUNICHAR\b': ('NCHAR', 'Changed UNICHAR to NCHAR'),
            r'\bUNIVARCHAR\b': ('NVARCHAR', 'Changed UNIVARCHAR to NVARCHAR'),
            r'\bUNITEXT\b': ('NVARCHAR(MAX)', 'Changed UNITEXT to NVARCHAR(MAX)'),
            r'\bBIT\b': ('BIT', 'Kept BIT (compatible)'),
            r'\bIMAGE\b': ('VARBINARY(MAX)', 'Changed IMAGE to VARBINARY(MAX)'),
            r'\bVARBINARY\b': ('VARBINARY', 'Kept VARBINARY (compatible)'),
            r'\bBINARY\b': ('BINARY', 'Kept BINARY (compatible)'),
            
            # Syntax conversions
            r'CREATE\s+PROCEDURE': ('CREATE PROCEDURE', 'Kept CREATE PROCEDURE (compatible)'),
            r'EXEC\s+sp_procxmode.*$': ('-- Removed sp_procxmode (not needed in MS SQL)', 'Removed sp_procxmode call'),
            r'\bGO\b': ('GO', 'Kept GO (compatible)'),
            r'set rowcount\s+(\d+)': ('SET ROWCOUNT \\1', 'Kept SET ROWCOUNT (compatible)'),
            r'set rowcount\s+0': ('SET ROWCOUNT 0', 'Kept SET ROWCOUNT 0 (compatible)'),
            
            # Function conversions
            r'getdate\(\)': ('GETDATE()', 'Kept GETDATE() (compatible)'),
            r'convert\(([^,]+),\s*([^)]+)\)': ('CONVERT(\\1, \\2)', 'Kept CONVERT (compatible)'),
            r'select\s+@@identity': ('SELECT SCOPE_IDENTITY()', 'Changed @@identity to SCOPE_IDENTITY()'),
            r'@@rowcount': ('@@ROWCOUNT', 'Kept @@ROWCOUNT (compatible)'),
            r'isnull\(': ('ISNULL(', 'Kept ISNULL (compatible)'),
            
            # Transaction handling
            r'begin\s+tran(saction)?': ('BEGIN TRANSACTION', 'Changed to standard BEGIN TRANSACTION'),
            r'commit\s+tran(saction)?': ('COMMIT TRANSACTION', 'Changed to standard COMMIT TRANSACTION'),
            r'rollback\s+tran(saction)?': ('ROLLBACK TRANSACTION', 'Changed to standard ROLLBACK TRANSACTION'),
            r'save\s+tran(saction)?': ('SAVE TRANSACTION', 'Changed to standard SAVE TRANSACTION'),
            
            # Temporary table handling
            r'create\s+table\s+#': ('CREATE TABLE #', 'Kept temporary table (compatible)'),
            r'truncate\s+table\s+#': ('TRUNCATE TABLE #', 'Kept truncate temporary table (compatible)'),
            
            # Remove Sybase-specific options
            r'WITH\s+RECOMPILE': ('-- WITH RECOMPILE', 'Commented out WITH RECOMPILE option'),
            
            # Handle HOLDLOCK hints
            r'HOLDLOCK': ('SERIALIZABLE', 'Changed HOLDLOCK to SERIALIZABLE hint'),
            
            # Handle different auto-increment syntax
            r'IDENTITY\s+\((\d+),\s*(\d+)\)': ('IDENTITY(\\1, \\2)', 'Kept IDENTITY (compatible)'),
            
            # Handle different NULL constraint syntax
            r'NULL\s+DEFAULT\s+NULL': ('NULL', 'Simplified NULL DEFAULT NULL to NULL'),
            
            # Update OUTPUT parameter syntax
            r'(\w+)\s+(\w+)\s+OUTPUT': ('\\1 \\2 OUTPUT', 'Kept OUTPUT parameter (compatible)'),
            
            # Update SELECT INTO syntax
            r'SELECT\s+.*\s+INTO\s+#': ('SELECT INTO #', 'Kept SELECT INTO (compatible)'),
            
            # Handle different date functions
            r'DATEADD\((\w+),\s*([^,]+),\s*([^)]+)\)': ('DATEADD(\\1, \\2, \\3)', 'Kept DATEADD (compatible)'),
            r'DATEDIFF\((\w+),\s*([^,]+),\s*([^)]+)\)': ('DATEDIFF(\\1, \\2, \\3)', 'Kept DATEDIFF (compatible)'),
            r'DATENAME\((\w+),\s*([^)]+)\)': ('DATENAME(\\1, \\2)', 'Kept DATENAME (compatible)'),
            r'DATEPART\((\w+),\s*([^)]+)\)': ('DATEPART(\\1, \\2)', 'Kept DATEPART (compatible)'),
            
            # Replace print with raiserror for error messages
            r'PRINT\s+\'ERROR:([^\']+)\'': ('RAISERROR(\'\\1\', 16, 1)', 'Changed ERROR PRINT to RAISERROR'),
            
            # Handle different TOP syntax
            r'TOP\s+(\d+)': ('TOP \\1', 'Kept TOP (compatible)'),
            
            # Convert Sybase @@error handling to TRY...CATCH
            r'IF\s+@@error\s*!=\s*0': ('IF @@ERROR != 0', 'Consider replacing with TRY...CATCH'),
            
            # Replace sa_license_check (Sybase-specific)
            r'sa_license_check\(.*\)': ('1 /* Removed sa_license_check - not needed in MS SQL */', 'Removed sa_license_check'),
        }

    def _get_oracle_rules(self):
        """Get conversion rules for Oracle target."""
        return {
            # Data type conversions
            r'\bINTEGER\b': ('NUMBER(10)', 'Changed INTEGER to NUMBER(10)'),
            r'\bSMALLINT\b': ('NUMBER(5)', 'Changed SMALLINT to NUMBER(5)'),
            r'\bTINYINT\b': ('NUMBER(3)', 'Changed TINYINT to NUMBER(3)'),
            r'\bDECIMAL\b': ('NUMBER', 'Changed DECIMAL to NUMBER'),
            r'\bNUMERIC\b': ('NUMBER', 'Changed NUMERIC to NUMBER'),
            r'\bMONEY\b': ('NUMBER(19,4)', 'Changed MONEY to NUMBER(19,4)'),
            r'\bSMALLMONEY\b': ('NUMBER(10,4)', 'Changed SMALLMONEY to NUMBER(10,4)'),
            r'\bFLOAT\b': ('FLOAT', 'Kept FLOAT (compatible)'),
            r'\bREAL\b': ('FLOAT', 'Changed REAL to FLOAT'),
            r'\bDATETIME\b': ('TIMESTAMP', 'Changed DATETIME to TIMESTAMP'),
            r'\bSMALLDATETIME\b': ('TIMESTAMP', 'Changed SMALLDATETIME to TIMESTAMP'),
            r'\bDATE\b': ('DATE', 'Kept DATE (compatible)'),
            r'\bTIME\b': ('TIMESTAMP', 'Changed TIME to TIMESTAMP'),
            r'\bCHAR\b': ('CHAR', 'Kept CHAR (compatible)'),
            r'\bVARCHAR\b': ('VARCHAR2', 'Changed VARCHAR to VARCHAR2'),
            r'\bTEXT\b': ('CLOB', 'Changed TEXT to CLOB'),
            r'\bUNICHAR\b': ('NCHAR', 'Changed UNICHAR to NCHAR'),
            r'\bUNIVARCHAR\b': ('NVARCHAR2', 'Changed UNIVARCHAR to NVARCHAR2'),
            r'\bUNITEXT\b': ('NCLOB', 'Changed UNITEXT to NCLOB'),
            r'\bBIT\b': ('NUMBER(1)', 'Changed BIT to NUMBER(1)'),
            r'\bIMAGE\b': ('BLOB', 'Changed IMAGE to BLOB'),
            r'\bVARBINARY\b': ('RAW', 'Changed VARBINARY to RAW'),
            r'\bBINARY\b': ('RAW', 'Changed BINARY to RAW'),
            
            # Syntax conversions
            r'CREATE\s+PROCEDURE': ('CREATE OR REPLACE PROCEDURE', 'Changed to Oracle CREATE OR REPLACE PROCEDURE'),
            r'EXEC\s+sp_procxmode.*$': ('-- Removed sp_procxmode (not needed in Oracle)', 'Removed sp_procxmode call'),
            r'\bGO\b': ('/', 'Changed GO to / for Oracle execution delimiter'),
            r'set rowcount\s+(\d+)': ('-- Oracle doesn\'t support direct rowcount limitation, use ROWNUM in queries instead', 'Commented out rowcount limitation'),
            r'set rowcount\s+0': ('-- Oracle doesn\'t need to reset rowcount', 'Commented out rowcount reset'),
            
            # Function conversions
            r'getdate\(\)': ('SYSDATE', 'Changed GETDATE() to SYSDATE'),
            r'convert\(([^,]+),\s*([^)]+)\)': ('CAST(\\2 AS \\1)', 'Changed CONVERT to CAST with reordered parameters'),
            r'select\s+@@identity': ('SELECT seq_name.CURRVAL FROM dual', 'Changed @@identity to sequence.CURRVAL (replace seq_name with your sequence)'),
            r'@@rowcount': ('SQL%ROWCOUNT', 'Changed @@ROWCOUNT to SQL%ROWCOUNT'),
            r'isnull\(([^,]+),\s*([^)]+)\)': ('NVL(\\1, \\2)', 'Changed ISNULL to NVL'),
            
            # Transaction handling
            r'begin\s+tran(saction)?': ('-- Transaction begins implicitly in Oracle', 'Commented out explicit transaction start'),
            r'commit\s+tran(saction)?': ('COMMIT', 'Changed to Oracle COMMIT'),
            r'rollback\s+tran(saction)?': ('ROLLBACK', 'Changed to Oracle ROLLBACK'),
            r'save\s+tran(saction)?': ('SAVEPOINT', 'Changed to Oracle SAVEPOINT'),
            
            # Temporary table handling
            r'create\s+table\s+#(\w+)': ('CREATE GLOBAL TEMPORARY TABLE \\1', 'Changed temporary table to Oracle global temporary table'),
            r'truncate\s+table\s+#(\w+)': ('TRUNCATE TABLE \\1', 'Changed temporary table reference for truncate'),
            
            # Handle different auto-increment syntax
            r'IDENTITY\s*\((\d+),\s*(\d+)\)': ('/* Use Oracle sequence instead: \nCREATE SEQUENCE seq_name START WITH \\1 INCREMENT BY \\2; \nThen use seq_name.NEXTVAL in your INSERT statements */', 
                                            'Replace IDENTITY with Oracle sequence'),
            
            # Handle NULL constraint syntax
            r'NULL\s+DEFAULT\s+NULL': ('NULL', 'Simplified NULL DEFAULT NULL to NULL'),
            
            # Update parameter syntax
            r'(\w+)\s+(\w+)\s+OUTPUT': ('\\2 OUT \\1', 'Changed OUTPUT parameter to Oracle OUT parameter with switched order'),
            
            # Replace SELECT INTO with Oracle's INSERT INTO ... SELECT
            r'SELECT\s+(.*)\s+INTO\s+([^FROM]+)': ('INSERT INTO \\2 SELECT \\1', 'Changed SELECT INTO to INSERT INTO ... SELECT'),
            
            # Handle different date functions
            r'DATEADD\((\w+),\s*([^,]+),\s*([^)]+)\)': ('/* Use appropriate Oracle date function: \nYEAR: ADD_MONTHS(\\3, 12*\\2) \nMONTH: ADD_MONTHS(\\3, \\2) \nDAY: \\3 + \\2 \nHOUR: \\3 + \\2/24 \nMINUTE: \\3 + \\2/(24*60) \nSECOND: \\3 + \\2/(24*60*60) */', 
                                                      'Replace DATEADD with Oracle date arithmetic'),
            r'DATEDIFF\((\w+),\s*([^,]+),\s*([^)]+)\)': ('/* Use appropriate Oracle date function: \nYEAR: MONTHS_BETWEEN(\\3, \\2)/12 \nMONTH: MONTHS_BETWEEN(\\3, \\2) \nDAY: \\3 - \\2 \nHOUR: (\\3 - \\2)*24 \nMINUTE: (\\3 - \\2)*24*60 \nSECOND: (\\3 - \\2)*24*60*60 */', 
                                                       'Replace DATEDIFF with Oracle date arithmetic'),
            
            # Replace PRINT with DBMS_OUTPUT
            r'PRINT\s+\'([^\']+)\'': ('DBMS_OUTPUT.PUT_LINE(\'\\1\')', 'Changed PRINT to DBMS_OUTPUT.PUT_LINE'),
            r'PRINT\s+\"([^\"]+)\"': ('DBMS_OUTPUT.PUT_LINE("\\1")', 'Changed PRINT to DBMS_OUTPUT.PUT_LINE'),
            
            # Handle different TOP syntax
            r'TOP\s+(\d+)': ('/* Use ROWNUM <= \\1 in WHERE clause or use: \nSELECT * FROM (YOUR_QUERY) WHERE ROWNUM <= \\1 */', 'Replace TOP with ROWNUM or ROW_NUMBER()'),
            
            # Convert error handling
            r'IF\s+@@error\s*!=\s*0': ('/* Replace with Oracle exception handling: \nBEGIN \n    -- Your code here \nEXCEPTION \n    WHEN OTHERS THEN \n        -- Error handling here \nEND; */', 
                                     'Replace error check with Oracle EXCEPTION block'),
            
            # Replace Sybase-specific functions
            r'sa_license_check\(.*\)': ('1 /* Removed sa_license_check - not needed in Oracle */', 'Removed sa_license_check'),
            
            # Replace square bracket identifiers with double quotes
            r'\[([^\]]+)\]': ('"\\1"', 'Changed square bracket identifiers to double quotes'),
            
            # Add semicolons to statement ends
            r'(INSERT|UPDATE|DELETE|SELECT)(.+?)(\n\s*[A-Z])': ('\\1\\2;\\3', 'Added semicolons to statement ends'),
        }

    def _get_java_rules(self):
        """Get conversion rules for Java target."""
        return {
            # Class structure for stored procedures
            r'CREATE\s+PROCEDURE\s+(\w+)\s*\((.*?)\)': 
                (
                    'public class \\1 {\n'
                    '    private Connection connection;\n\n'
                    '    public \\1(Connection connection) {\n'
                    '        this.connection = connection;\n'
                    '    }\n\n'
                    '    public void execute(\\2) throws SQLException {\n'
                    '        // Procedure implementation\n'
                    '    }',
                    'Converted stored procedure to Java class'
                ),
            
            # Parameter conversions
            r'@(\w+)\s+(\w+)(?:\((\d+)(?:,\s*(\d+))?\))?': 
                (
                    lambda m: self._convert_param_to_java(m.group(1), m.group(2), m.group(3), m.group(4)),
                    'Converted procedure parameter to Java'
                ),
            
            # Variable declarations
            r'DECLARE\s+@(\w+)\s+(\w+)(?:\((\d+)(?:,\s*(\d+))?\))?': 
                (
                    lambda m: f"{self._sql_type_to_java(m.group(2), m.group(3), m.group(4))} {m.group(1)};",
                    'Converted variable declaration to Java'
                ),
            
            # SET statements
            r'SET\s+@(\w+)\s*=\s*(.+)': 
                (
                    '\\1 = \\2;',
                    'Converted SET statement to Java assignment'
                ),
            
            # SELECT statements
            r'SELECT\s+(.*?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?': 
                (
                    lambda m: f'String sql = "SELECT {m.group(1)} FROM {m.group(2)}{" WHERE " + m.group(3) if m.group(3) else ""}";\n'
                    f'try (PreparedStatement stmt = connection.prepareStatement(sql)) {{\n'
                    f'    // Set parameters if needed\n'
                    f'    try (ResultSet rs = stmt.executeQuery()) {{\n'
                    f'        while (rs.next()) {{\n'
                    f'            // Process results\n'
                    f'            // Example: String value = rs.getString("column_name");\n'
                    f'        }}\n'
                    f'    }}\n'
                    f'}}',
                    'Converted SELECT to Java JDBC query'
                ),
            
            # INSERT statements
            r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)': 
                (
                    lambda m: f'String sql = "INSERT INTO {m.group(1)} ({m.group(2)}) VALUES ({", ".join(["?"] * len(m.group(3).split(",")))})"; \n'
                    f'try (PreparedStatement stmt = connection.prepareStatement(sql)) {{\n'
                    f'    // Set parameters\n'
                    f'    // Example: stmt.setString(1, value);\n'
                    f'    stmt.executeUpdate();\n'
                    f'}}',
                    'Converted INSERT to Java JDBC statement'
                ),
            
            # UPDATE statements
            r'UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+?))?': 
                (
                    lambda m: f'String sql = "UPDATE {m.group(1)} SET {m.group(2)}{" WHERE " + m.group(3) if m.group(3) else ""}";\n'
                    f'try (PreparedStatement stmt = connection.prepareStatement(sql)) {{\n'
                    f'    // Set parameters if needed\n'
                    f'    stmt.executeUpdate();\n'
                    f'}}',
                    'Converted UPDATE to Java JDBC statement'
                ),
            
            # DELETE statements
            r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?': 
                (
                    lambda m: f'String sql = "DELETE FROM {m.group(1)}{" WHERE " + m.group(2) if m.group(2) else ""}";\n'
                    f'try (PreparedStatement stmt = connection.prepareStatement(sql)) {{\n'
                    f'    // Set parameters if needed\n'
                    f'    stmt.executeUpdate();\n'
                    f'}}',
                    'Converted DELETE to Java JDBC statement'
                ),
            
            # IF statements
            r'IF\s+(.+?)\s+BEGIN\s+(.*?)\s+END': 
                (
                    'if (\\1) {\n    \\2\n}',
                    'Converted IF-BEGIN-END block to Java if block'
                ),
            
            # WHILE loops
            r'WHILE\s+(.+?)\s+BEGIN\s+(.*?)\s+END': 
                (
                    'while (\\1) {\n    \\2\n}',
                    'Converted WHILE loop to Java while loop'
                ),
            
            # Transaction management
            r'BEGIN\s+TRANSACTION|BEGIN\s+TRAN': 
                (
                    'connection.setAutoCommit(false);',
                    'Set up transaction in Java JDBC'
                ),
            
            r'COMMIT\s+TRANSACTION|COMMIT\s+TRAN': 
                (
                    'connection.commit();\nconnection.setAutoCommit(true);',
                    'Commit transaction in Java JDBC'
                ),
            
            r'ROLLBACK\s+TRANSACTION|ROLLBACK\s+TRAN': 
                (
                    'connection.rollback();\nconnection.setAutoCommit(true);',
                    'Rollback transaction in Java JDBC'
                ),
            
            # SQL functions
            r'GETDATE\(\)': 
                (
                    'new java.sql.Timestamp(System.currentTimeMillis())',
                    'Converted GETDATE() to Java Timestamp'
                ),
            
            r'UPPER\(([^)]+)\)': 
                (
                    '\\1.toUpperCase()',
                    'Converted UPPER() to Java toUpperCase()'
                ),
            
            r'LOWER\(([^)]+)\)': 
                (
                    '\\1.toLowerCase()',
                    'Converted LOWER() to Java toLowerCase()'
                ),
            
            r'LEN\(([^)]+)\)': 
                (
                    '\\1.length()',
                    'Converted LEN() to Java length()'
                ),
            
            r'SUBSTRING\(([^,]+),\s*([^,]+),\s*([^)]+)\)': 
                (
                    '\\1.substring(\\2 - 1, \\2 - 1 + \\3)',
                    'Converted SUBSTRING() to Java substring() (adjusted for 0-based indexing)'
                ),
            
            r'ISNULL\(([^,]+),\s*([^)]+)\)': 
                (
                    '(\\1 != null ? \\1 : \\2)',
                    'Converted ISNULL() to Java ternary operator'
                ),
            
            # Comments
            r'--(.*)': 
                (
                    '// \\1',
                    'Converted SQL comment to Java comment'
                ),
            
            r'/\*(.*?)\*/': 
                (
                    '/* \\1 */',
                    'Kept multi-line comment'
                ),
            
            # GO command
            r'\bGO\b': 
                (
                    '// End of batch',
                    'Removed GO statement (not needed in Java)'
                ),
        }
    
    def _sql_type_to_java(self, sql_type, size=None, precision=None):
        """Convert SQL data type to appropriate Java data type."""
        sql_type = sql_type.upper()
        
        # Map of SQL types to Java types
        type_map = {
            'INTEGER': 'int',
            'INT': 'int',
            'SMALLINT': 'short',
            'TINYINT': 'byte',
            'BIGINT': 'long',
            'DECIMAL': 'java.math.BigDecimal',
            'NUMERIC': 'java.math.BigDecimal',
            'FLOAT': 'double',
            'REAL': 'float',
            'MONEY': 'java.math.BigDecimal',
            'SMALLMONEY': 'java.math.BigDecimal',
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
            'DATETIME': 'java.sql.Timestamp',
            'SMALLDATETIME': 'java.sql.Timestamp',
            'DATE': 'java.sql.Date',
            'TIME': 'java.sql.Time',
            'UNIQUEIDENTIFIER': 'java.util.UUID',
        }
        
        return type_map.get(sql_type, 'Object')
    
    def _convert_param_to_java(self, name, sql_type, size=None, precision=None):
        """Convert a SQL parameter to a Java method parameter."""
        java_type = self._sql_type_to_java(sql_type, size, precision)
        return f"{java_type} {name}"
    
    def _get_javascript_rules(self):
        """Get conversion rules specifically for JavaScript (client-side focus)."""
        js_rules = {
            # Class structure for stored procedures
            r'CREATE\s+PROCEDURE\s+(\w+)\s*\((.*?)\)': 
                (
                    'class \\1 {\n'
                    '  constructor(dbClient) {\n'
                    '    this.dbClient = dbClient;\n'
                    '  }\n\n'
                    '  async execute(\\2) {\n'
                    '    // Procedure implementation\n'
                    '  }\n'
                    '}',
                    'Converted stored procedure to JavaScript class'
                ),
            
            # Parameter conversions
            r'@(\w+)\s+(\w+)(?:\((\d+)(?:,\s*(\d+))?\))?': 
                (
                    '\\1', 
                    'Converted parameter name to JavaScript parameter'
                ),
            
            # Variable declarations
            r'DECLARE\s+@(\w+)\s+(\w+)(?:\((\d+)(?:,\s*(\d+))?\))?': 
                (
                    'let \\1;', 
                    'Converted variable declaration to JavaScript'
                ),
            
            # SET statements
            r'SET\s+@(\w+)\s*=\s*(.+)': 
                (
                    '\\1 = \\2;',
                    'Converted SET statement to JavaScript assignment'
                ),
            
            # Comments
            r'--(.*)': 
                (
                    '// \\1',
                    'Converted SQL comment to JavaScript comment'
                ),
            
            # GO command
            r'\bGO\b': 
                (
                    '// End of statement block',
                    'Removed GO command (not needed in JavaScript)'
                ),
        }
        
        return js_rules

    def convert_file(self, input_file, output_file):
        """
        Converts a single Sybase stored procedure file to the target database format.
        
        Args:
            input_file: Path to the input Sybase procedure file
            output_file: Path to write the converted procedure
            
        Returns:
            int: Number of changes made during conversion
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            changes_made = 0
            
            # Add header comment to indicate this is a converted file
            header = f"""/*
=============================================
Converted from Sybase to {self.target_db.upper()}
Original file: {os.path.basename(input_file)}
Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=============================================
*/

"""
            # For Java, change the file extension to .java and add additional info
            if self.target_db == 'java':
                output_file = output_file.with_suffix('.java')
                header += """/*
NOTE: This is a Java file for JDBC. SQL stored procedures have been 
converted to Java classes that can be executed using JDBC.
Many complex SQL features may require manual adjustment.
*/

"""
            # For JavaScript, change the file extension to .js and add additional info
            if self.target_db == 'javascript':
                output_file = output_file.with_suffix('.js')
                header += """/*
NOTE: This is a JavaScript file. SQL stored procedures have been 
converted to JavaScript classes.
Many complex SQL features may require manual adjustment.
*/

"""

            # Apply each conversion rule
            for pattern, (replacement, comment) in self.conversion_rules.items():
                # Count matches before replacement
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    # Apply the replacement
                    if callable(replacement):
                        # If replacement is a lambda function, apply it with match objects
                        updated_content = re.sub(
                            pattern,
                            lambda m: f"{replacement(m)} /* {comment} */",
                            content,
                            flags=re.IGNORECASE
                        )
                    else:
                        # Otherwise apply the string replacement
                        updated_content = re.sub(
                            pattern,
                            lambda m: f"{replacement} /* {comment} */",
                            content,
                            flags=re.IGNORECASE
                        )
                    content = updated_content
                    changes_made += matches
                    self.logger.info(f"Applied rule '{pattern}' to '{replacement}' ({matches} occurrences) in {os.path.basename(input_file)}")
            
            # Write converted content to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(header + content)
                
            if changes_made > 0:
                self.logger.info(f"Converted {os.path.basename(input_file)} with {changes_made} changes")
            else:
                self.logger.info(f"No changes needed for {os.path.basename(input_file)}")
                
            return changes_made
                
        except Exception as e:
            self.logger.error(f"Error converting {input_file}: {str(e)}")
            return 0

    def process_directory(self, input_dir, output_dir):
        """
        Processes all files in the input directory and converts them to the target database format.
        
        Args:
            input_dir: Directory containing Sybase stored procedure files
            output_dir: Directory to write converted procedures
            
        Returns:
            tuple: (files_processed, files_with_changes, total_changes)
        """
        input_path = Path(input_dir)
        # Create a target-specific subdirectory in the output path
        output_path = Path(output_dir) / self.target_db
        
        # Make sure output directory exists
        output_path.mkdir(exist_ok=True, parents=True)
        
        files_processed = 0
        files_with_changes = 0
        total_changes = 0
        
        self.logger.info(f"Starting conversion from {input_dir} to {output_path} for target {self.target_db}")
        
        # Process all .sql files in input directory
        file_pattern = '**/*.sql'
        for file_path in input_path.glob(file_pattern):
            # Determine the relative path to maintain directory structure
            rel_path = file_path.relative_to(input_path)
            output_file = output_path / rel_path
            
            # Create subdirectories if needed
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Convert the file
            changes = self.convert_file(file_path, output_file)
            
            files_processed += 1
            if changes > 0:
                files_with_changes += 1
                total_changes += changes
                
        self.logger.info(f"Conversion complete. Processed {files_processed} files, {files_with_changes} had changes, {total_changes} total changes.")
        return (files_processed, files_with_changes, total_changes)


def setup_logging():
    """
    Sets up logging to both console and file.
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger('sybase_converter')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler with timestamped log file
    log_file = f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    parser = argparse.ArgumentParser(description='Convert Sybase stored procedures to other database formats.')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing Sybase stored procedures')
    parser.add_argument('--output', '-o', required=True, help='Output directory for converted procedures')
    parser.add_argument('--target', '-t', choices=['mssql', 'oracle', 'java', 'javascript'], default='mssql', 
                        help='Target database type (default: mssql)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    logger.info(f"Starting Sybase to {args.target.upper()} conversion")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Target database: {args.target}")
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        logger.error(f"Input directory {args.input} does not exist or is not a directory")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logger.info(f"Created output directory: {args.output}")
    
    # Initialize converter
    converter = SybaseConverter(logger, args.target)
    
    # Process the directory
    files_processed, files_with_changes, total_changes = converter.process_directory(args.input, args.output)
    
    logger.info("Conversion completed")
    logger.info(f"Files processed: {files_processed}")
    logger.info(f"Files with changes: {files_with_changes}")
    logger.info(f"Total changes made: {total_changes}")
    
    return 0


if __name__ == "__main__":
    exit(main())
