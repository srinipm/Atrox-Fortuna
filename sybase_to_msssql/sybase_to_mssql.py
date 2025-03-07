#!/usr/bin/env python3
"""
Sybase to MS SQL Stored Procedure Converter

This script converts Sybase stored procedures to MS SQL format.
It reads files from an input directory, applies conversion rules,
and writes the converted procedures to an output directory.
Changes are logged to a log file, and comments are added
to mark changes from the original code.

Usage:
    python sybase_to_mssql.py --input <input_dir> --output <output_dir>
"""

import os
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path


class SybaseToMSSQLConverter:
    """Converts Sybase stored procedures to MS SQL format."""

    def __init__(self, logger):
        self.logger = logger
        # Dictionary of conversion patterns
        # Format: 'regex_pattern': ('replacement', 'comment')
        self.conversion_rules = {
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

    def convert_file(self, input_file, output_file):
        """
        Converts a single Sybase stored procedure file to MS SQL format.
        
        Args:
            input_file: Path to the input Sybase procedure file
            output_file: Path to write the converted MS SQL procedure
            
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
Converted from Sybase to MS SQL
Original file: {os.path.basename(input_file)}
Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=============================================
*/

"""
            # Apply each conversion rule
            for pattern, (replacement, comment) in self.conversion_rules.items():
                # Count matches before replacement
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    # Apply the replacement
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
        Processes all files in the input directory and converts them to MS SQL format.
        
        Args:
            input_dir: Directory containing Sybase stored procedure files
            output_dir: Directory to write converted MS SQL procedures
            
        Returns:
            tuple: (files_processed, files_with_changes, total_changes)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Make sure output directory exists
        output_path.mkdir(exist_ok=True, parents=True)
        
        files_processed = 0
        files_with_changes = 0
        total_changes = 0
        
        self.logger.info(f"Starting conversion from {input_dir} to {output_dir}")
        
        # Process all .sql files in input directory
        for file_path in input_path.glob('**/*.sql'):
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
    logger = logging.getLogger('sybase_to_mssql')
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
    parser = argparse.ArgumentParser(description='Convert Sybase stored procedures to MS SQL format.')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing Sybase stored procedures')
    parser.add_argument('--output', '-o', required=True, help='Output directory for converted MS SQL procedures')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    logger.info("Starting Sybase to MS SQL stored procedure conversion")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        logger.error(f"Input directory {args.input} does not exist or is not a directory")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logger.info(f"Created output directory: {args.output}")
    
    # Initialize converter
    converter = SybaseToMSSQLConverter(logger)
    
    # Process the directory
    files_processed, files_with_changes, total_changes = converter.process_directory(args.input, args.output)
    
    logger.info("Conversion completed")
    logger.info(f"Files processed: {files_processed}")
    logger.info(f"Files with changes: {files_with_changes}")
    logger.info(f"Total changes made: {total_changes}")
    
    return 0


if __name__ == "__main__":
    exit(main())