#!/usr/bin/env python3
"""
Sybase Database Converter

This script scans Sybase database system tables and converts database objects
to equivalent structures in Microsoft SQL Server, Oracle, or MongoDB.
It also generates data migration scripts for the target database.

Usage:
  python sybase_converter.py --source-conn "user/password@server:port/dbname" 
                            --target-type [sqlserver|oracle|mongodb] 
                            --target-conn "connection_string"
                            [--output-dir OUTPUT_DIR]
                            [--include-data]
                            [--objects OBJECT_LIST]
                            [--exclude-objects EXCLUDE_LIST]
                            [--batch-size BATCH_SIZE]
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set, Optional

# Database connectors
import pymssql  # For Sybase and SQL Server
import cx_Oracle  # For Oracle
import pymongo  # For MongoDB

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sybase_converter.log')
    ]
)
logger = logging.getLogger('sybase_converter')

# Constants for object types
TABLE = 'TABLE'
VIEW = 'VIEW'
PROCEDURE = 'PROCEDURE'
FUNCTION = 'FUNCTION'
TRIGGER = 'TRIGGER'
INDEX = 'INDEX'
CONSTRAINT = 'CONSTRAINT'
SEQUENCE = 'SEQUENCE'
DATATYPE_MAPPING = {
    # Sybase to SQL Server
    'sqlserver': {
        'char': 'char',
        'varchar': 'varchar',
        'text': 'text',
        'int': 'int',
        'smallint': 'smallint',
        'tinyint': 'tinyint',
        'bigint': 'bigint',
        'decimal': 'decimal',
        'numeric': 'numeric',
        'float': 'float',
        'real': 'real',
        'money': 'money',
        'smallmoney': 'smallmoney',
        'datetime': 'datetime',
        'smalldatetime': 'smalldatetime',
        'timestamp': 'timestamp',
        'binary': 'binary',
        'varbinary': 'varbinary',
        'image': 'image',
        'bit': 'bit',
        'unichar': 'nchar',
        'univarchar': 'nvarchar',
        'unitext': 'ntext',
        'date': 'date',
        'time': 'time',
    },
    # Sybase to Oracle
    'oracle': {
        'char': 'CHAR',
        'varchar': 'VARCHAR2',
        'text': 'CLOB',
        'int': 'NUMBER(10)',
        'smallint': 'NUMBER(5)',
        'tinyint': 'NUMBER(3)',
        'bigint': 'NUMBER(19)',
        'decimal': 'NUMBER',
        'numeric': 'NUMBER',
        'float': 'FLOAT',
        'real': 'FLOAT',
        'money': 'NUMBER(19,4)',
        'smallmoney': 'NUMBER(10,4)',
        'datetime': 'TIMESTAMP',
        'smalldatetime': 'TIMESTAMP',
        'timestamp': 'RAW(8)',
        'binary': 'RAW',
        'varbinary': 'BLOB',
        'image': 'BLOB',
        'bit': 'NUMBER(1)',
        'unichar': 'NCHAR',
        'univarchar': 'NVARCHAR2',
        'unitext': 'NCLOB',
        'date': 'DATE',
        'time': 'TIMESTAMP',
    },
    # Sybase to MongoDB JSON Schema types
    'mongodb': {
        'char': {'bsonType': 'string'},
        'varchar': {'bsonType': 'string'},
        'text': {'bsonType': 'string'},
        'int': {'bsonType': 'int'},
        'smallint': {'bsonType': 'int'},
        'tinyint': {'bsonType': 'int'},
        'bigint': {'bsonType': 'long'},
        'decimal': {'bsonType': 'decimal'},
        'numeric': {'bsonType': 'decimal'},
        'float': {'bsonType': 'double'},
        'real': {'bsonType': 'double'},
        'money': {'bsonType': 'decimal'},
        'smallmoney': {'bsonType': 'decimal'},
        'datetime': {'bsonType': 'date'},
        'smalldatetime': {'bsonType': 'date'},
        'timestamp': {'bsonType': 'timestamp'},
        'binary': {'bsonType': 'binData'},
        'varbinary': {'bsonType': 'binData'},
        'image': {'bsonType': 'binData'},
        'bit': {'bsonType': 'bool'},
        'unichar': {'bsonType': 'string'},
        'univarchar': {'bsonType': 'string'},
        'unitext': {'bsonType': 'string'},
        'date': {'bsonType': 'date'},
        'time': {'bsonType': 'date'},
    }
}

class SybaseScanner:
    """Scans Sybase system tables to extract database object metadata."""
    
    def __init__(self, connection_string: str):
        """
        Initialize the Sybase scanner with connection string.
        
        Args:
            connection_string: Format "user/password@server:port/dbname"
        """
        self.conn_string = connection_string
        self.conn = None
        self.cursor = None
        self.metadata = {
            TABLE: {},
            VIEW: {},
            PROCEDURE: {},
            FUNCTION: {},
            TRIGGER: {},
            INDEX: {},
            CONSTRAINT: {},
            SEQUENCE: {},
        }
        
    def parse_connection_string(self) -> Tuple[str, str, str, int, str]:
        """Parse the connection string into components."""
        user_pass, host_port_db = self.conn_string.split('@')
        user, password = user_pass.split('/')
        
        if '/' in host_port_db:
            host_port, db = host_port_db.split('/')
        else:
            host_port, db = host_port_db, ''
            
        if ':' in host_port:
            host, port = host_port.split(':')
            port = int(port)
        else:
            host, port = host_port, 5000  # Default Sybase port
            
        return user, password, host, port, db
    
    def connect(self) -> None:
        """Establish connection to the Sybase database."""
        try:
            user, password, host, port, db = self.parse_connection_string()
            self.conn = pymssql.connect(
                server=host,
                user=user,
                password=password,
                database=db,
                port=port,
                tds_version='5.0'  # Use TDS 5.0 for Sybase ASE
            )
            self.cursor = self.conn.cursor(as_dict=True)
            logger.info(f"Connected to Sybase database: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Failed to connect to Sybase: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Closed Sybase database connection")
    
    def scan_tables(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get table definitions."""
        logger.info("Scanning tables...")
        
        # Get table list
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as table_name,
                o.id as table_id,
                o.crdate as creation_date
            FROM 
                sysobjects o
                JOIN sysusers u ON o.uid = u.uid
            WHERE 
                o.type = 'U'
            ORDER BY 
                u.name, o.name
        """)
        tables = self.cursor.fetchall()
        
        for table in tables:
            table_id = table['table_id']
            owner = table['owner']
            table_name = table['table_name']
            fullname = f"{owner}.{table_name}"
            
            # Get column information
            self.cursor.execute("""
                SELECT 
                    c.name as column_name,
                    t.name as data_type,
                    c.length,
                    c.prec as precision,
                    c.scale,
                    CASE WHEN c.status & 8 = 8 THEN 1 ELSE 0 END as is_identity,
                    CASE WHEN c.status & 16 = 16 THEN 0 ELSE 1 END as is_nullable,
                    c.colid as column_id
                FROM 
                    syscolumns c
                    JOIN systypes t ON c.usertype = t.usertype
                WHERE 
                    c.id = %d
                ORDER BY 
                    c.colid
            """ % table_id)
            columns = self.cursor.fetchall()
            
            # Get primary key
            self.cursor.execute("""
                SELECT 
                    i.name as index_name,
                    index_col(o.name, i.indid, ic.colid) as column_name
                FROM 
                    sysindexes i
                    JOIN sysobjects o ON i.id = o.id
                    JOIN sysindexkeys ic ON i.id = ic.id AND i.indid = ic.indid
                WHERE 
                    o.id = %d
                    AND i.status & 2048 = 2048
                ORDER BY 
                    ic.colid
            """ % table_id)
            pk_columns = [row['column_name'] for row in self.cursor.fetchall()]
            
            # Get foreign keys
            self.cursor.execute("""
                SELECT 
                    o.name as constraint_name,
                    c.name as column_name,
                    ro.name as ref_table,
                    rc.name as ref_column
                FROM 
                    sysconstraints cn
                    JOIN sysobjects o ON cn.constrid = o.id
                    JOIN sysreferences r ON r.constrid = o.id
                    JOIN sysobjects po ON r.tableid = po.id
                    JOIN syscolumns c ON r.tableid = c.id AND r.fokey1 = c.colid
                    JOIN sysobjects ro ON r.reftabid = ro.id
                    JOIN syscolumns rc ON r.reftabid = rc.id AND r.refkey1 = rc.colid
                WHERE 
                    po.id = %d
                    AND o.type = 'RI'
            """ % table_id)
            foreign_keys = self.cursor.fetchall()
            
            # Store table metadata
            self.metadata[TABLE][fullname] = {
                'name': table_name,
                'owner': owner,
                'columns': columns,
                'primary_key': pk_columns,
                'foreign_keys': foreign_keys,
                'creation_date': table['creation_date'],
            }
            
        logger.info(f"Found {len(self.metadata[TABLE])} tables")
        return self.metadata[TABLE]
    
    def scan_views(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get view definitions."""
        logger.info("Scanning views...")
        
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as view_name,
                o.id as view_id,
                o.crdate as creation_date
            FROM 
                sysobjects o
                JOIN sysusers u ON o.uid = u.uid
            WHERE 
                o.type = 'V'
            ORDER BY 
                u.name, o.name
        """)
        views = self.cursor.fetchall()
        
        for view in views:
            view_id = view['view_id']
            owner = view['owner']
            view_name = view['view_name']
            fullname = f"{owner}.{view_name}"
            
            # Get view definition
            self.cursor.execute("""
                SELECT text
                FROM syscomments
                WHERE id = %d
                ORDER BY colid
            """ % view_id)
            text_parts = [row['text'] for row in self.cursor.fetchall()]
            view_definition = ''.join(text_parts)
            
            # Get column information
            self.cursor.execute("""
                SELECT 
                    c.name as column_name,
                    t.name as data_type,
                    c.length,
                    c.prec as precision,
                    c.scale,
                    CASE WHEN c.status & 8 = 8 THEN 1 ELSE 0 END as is_identity,
                    CASE WHEN c.status & 16 = 16 THEN 0 ELSE 1 END as is_nullable,
                    c.colid as column_id
                FROM 
                    syscolumns c
                    JOIN systypes t ON c.usertype = t.usertype
                WHERE 
                    c.id = %d
                ORDER BY 
                    c.colid
            """ % view_id)
            columns = self.cursor.fetchall()
            
            # Store view metadata
            self.metadata[VIEW][fullname] = {
                'name': view_name,
                'owner': owner,
                'columns': columns,
                'definition': view_definition,
                'creation_date': view['creation_date'],
            }
            
        logger.info(f"Found {len(self.metadata[VIEW])} views")
        return self.metadata[VIEW]
    
    def scan_procedures(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get stored procedure definitions."""
        logger.info("Scanning stored procedures...")
        
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as proc_name,
                o.id as proc_id,
                o.crdate as creation_date
            FROM 
                sysobjects o
                JOIN sysusers u ON o.uid = u.uid
            WHERE 
                o.type = 'P'
            ORDER BY 
                u.name, o.name
        """)
        procs = self.cursor.fetchall()
        
        for proc in procs:
            proc_id = proc['proc_id']
            owner = proc['owner']
            proc_name = proc['proc_name']
            fullname = f"{owner}.{proc_name}"
            
            # Get procedure definition
            self.cursor.execute("""
                SELECT text
                FROM syscomments
                WHERE id = %d
                ORDER BY colid
            """ % proc_id)
            text_parts = [row['text'] for row in self.cursor.fetchall()]
            proc_definition = ''.join(text_parts)
            
            # Store procedure metadata
            self.metadata[PROCEDURE][fullname] = {
                'name': proc_name,
                'owner': owner,
                'definition': proc_definition,
                'creation_date': proc['creation_date'],
            }
            
        logger.info(f"Found {len(self.metadata[PROCEDURE])} stored procedures")
        return self.metadata[PROCEDURE]
    
    def scan_functions(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get function definitions."""
        logger.info("Scanning functions...")
        
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as func_name,
                o.id as func_id,
                o.crdate as creation_date
            FROM 
                sysobjects o
                JOIN sysusers u ON o.uid = u.uid
            WHERE 
                o.type = 'SF'
            ORDER BY 
                u.name, o.name
        """)
        funcs = self.cursor.fetchall()
        
        for func in funcs:
            func_id = func['func_id']
            owner = func['owner']
            func_name = func['func_name']
            fullname = f"{owner}.{func_name}"
            
            # Get function definition
            self.cursor.execute("""
                SELECT text
                FROM syscomments
                WHERE id = %d
                ORDER BY colid
            """ % func_id)
            text_parts = [row['text'] for row in self.cursor.fetchall()]
            func_definition = ''.join(text_parts)
            
            # Store function metadata
            self.metadata[FUNCTION][fullname] = {
                'name': func_name,
                'owner': owner,
                'definition': func_definition,
                'creation_date': func['creation_date'],
            }
            
        logger.info(f"Found {len(self.metadata[FUNCTION])} functions")
        return self.metadata[FUNCTION]
    
    def scan_triggers(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get trigger definitions."""
        logger.info("Scanning triggers...")
        
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as trigger_name,
                o.id as trigger_id,
                o.crdate as creation_date,
                parent_obj.name as table_name,
                parent_user.name as table_owner
            FROM 
                sysobjects o
                JOIN sysusers u ON o.uid = u.uid
                JOIN sysobjects parent_obj ON o.deltrig = parent_obj.id
                JOIN sysusers parent_user ON parent_obj.uid = parent_user.uid
            WHERE 
                o.type = 'TR'
            ORDER BY 
                u.name, o.name
        """)
        triggers = self.cursor.fetchall()
        
        for trigger in triggers:
            trigger_id = trigger['trigger_id']
            owner = trigger['owner']
            trigger_name = trigger['trigger_name']
            fullname = f"{owner}.{trigger_name}"
            
            # Get trigger definition
            self.cursor.execute("""
                SELECT text
                FROM syscomments
                WHERE id = %d
                ORDER BY colid
            """ % trigger_id)
            text_parts = [row['text'] for row in self.cursor.fetchall()]
            trigger_definition = ''.join(text_parts)
            
            # Store trigger metadata
            self.metadata[TRIGGER][fullname] = {
                'name': trigger_name,
                'owner': owner,
                'table_name': trigger['table_name'],
                'table_owner': trigger['table_owner'],
                'definition': trigger_definition,
                'creation_date': trigger['creation_date'],
            }
            
        logger.info(f"Found {len(self.metadata[TRIGGER])} triggers")
        return self.metadata[TRIGGER]
    
    def scan_indexes(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get index definitions."""
        logger.info("Scanning indexes...")
        
        # Get all indexes that are not primary keys or foreign keys
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as table_name,
                i.name as index_name,
                i.status as index_status,
                i.indid as index_id
            FROM 
                sysindexes i
                JOIN sysobjects o ON i.id = o.id
                JOIN sysusers u ON o.uid = u.uid
            WHERE 
                o.type = 'U'
                AND i.indid > 0
                AND i.indid < 255
                AND NOT (i.status & 2048 = 2048)  -- Not a primary key
            ORDER BY 
                u.name, o.name, i.name
        """)
        indexes = self.cursor.fetchall()
        
        for index in indexes:
            table_name = index['table_name']
            owner = index['owner']
            index_name = index['index_name']
            index_id = index['index_id']
            is_unique = (index['index_status'] & 2) == 2
            
            # Get index columns
            self.cursor.execute("""
                SELECT 
                    index_col('%s', %d, %d) as column_name,
                    %d as key_ordinal
                FROM 
                    sysobjects
                WHERE 
                    id = object_id('%s')
            """ % (table_name, index_id, 1, 1, table_name))
            columns = []
            column = self.cursor.fetchone()
            
            if column and column['column_name']:
                columns.append(column['column_name'])
                
                # Check for additional columns
                for i in range(2, 17):  # Sybase allows up to 16 columns per index
                    self.cursor.execute("""
                        SELECT 
                            index_col('%s', %d, %d) as column_name,
                            %d as key_ordinal
                        FROM 
                            sysobjects
                        WHERE 
                            id = object_id('%s')
                    """ % (table_name, index_id, i, i, table_name))
                    column = self.cursor.fetchone()
                    if column and column['column_name']:
                        columns.append(column['column_name'])
                    else:
                        break
            
            fullname = f"{owner}.{table_name}.{index_name}"
            
            # Store index metadata
            self.metadata[INDEX][fullname] = {
                'name': index_name,
                'table_name': table_name,
                'owner': owner,
                'is_unique': is_unique,
                'columns': columns,
            }
            
        logger.info(f"Found {len(self.metadata[INDEX])} indexes")
        return self.metadata[INDEX]
    
    def scan_constraints(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get constraint definitions."""
        logger.info("Scanning constraints...")
        
        # Get check constraints
        self.cursor.execute("""
            SELECT 
                u.name as owner,
                o.name as table_name,
                co.name as constraint_name,
                co.id as constraint_id
            FROM 
                sysconstraints c
                JOIN sysobjects o ON c.tableid = o.id
                JOIN sysusers u ON o.uid = u.uid
                JOIN sysobjects co ON c.constrid = co.id
            WHERE 
                co.type = 'C'
            ORDER BY 
                u.name, o.name, co.name
        """)
        constraints = self.cursor.fetchall()
        
        for constraint in constraints:
            constraint_id = constraint['constraint_id']
            owner = constraint['owner']
            table_name = constraint['table_name']
            constraint_name = constraint['constraint_name']
            
            # Get constraint definition
            self.cursor.execute("""
                SELECT text
                FROM syscomments
                WHERE id = %d
                ORDER BY colid
            """ % constraint_id)
            text_parts = [row['text'] for row in self.cursor.fetchall()]
            constraint_definition = ''.join(text_parts)
            
            fullname = f"{owner}.{table_name}.{constraint_name}"
            
            # Store constraint metadata
            self.metadata[CONSTRAINT][fullname] = {
                'name': constraint_name,
                'table_name': table_name,
                'owner': owner,
                'type': 'CHECK',
                'definition': constraint_definition,
            }
            
        logger.info(f"Found {len(self.metadata[CONSTRAINT])} constraints")
        return self.metadata[CONSTRAINT]
    
    def scan_sequences(self) -> Dict[str, Any]:
        """Scan Sybase system tables to get sequence/auto-identity information."""
        logger.info("Scanning sequences (identity columns)...")
        
        # In Sybase, sequences are typically implemented as identity columns
        # We'll scan all tables for identity columns
        sequences = {}
        
        for table_fullname, table_meta in self.metadata[TABLE].items():
            for column in table_meta['columns']:
                if column['is_identity'] == 1:
                    sequence_name = f"{table_meta['name']}_{column['column_name']}_seq"
                    fullname = f"{table_meta['owner']}.{sequence_name}"
                    
                    # Get current identity value and increment
                    self.cursor.execute(f"""
                        SELECT
                            IDENT_SEED('{table_meta['owner']}.{table_meta['name']}') as seed,
                            IDENT_INCR('{table_meta['owner']}.{table_meta['name']}') as increment
                    """)
                    ident_info = self.cursor.fetchone()
                    
                    # Store sequence metadata
                    self.metadata[SEQUENCE][fullname] = {
                        'name': sequence_name,
                        'owner': table_meta['owner'],
                        'data_type': column['data_type'],
                        'start_value': ident_info['seed'] if ident_info else 1,
                        'increment_by': ident_info['increment'] if ident_info else 1,
                        'table_name': table_meta['name'],
                        'column_name': column['column_name'],
                    }
        
        logger.info(f"Found {len(self.metadata[SEQUENCE])} sequences (identity columns)")
        return self.metadata[SEQUENCE]
    
    def scan_all(self) -> Dict[str, Dict[str, Any]]:
        """Scan all database objects."""
        try:
            self.connect()
            
            self.scan_tables()
            self.scan_views()
            self.scan_procedures()
            self.scan_functions()
            self.scan_triggers()
            self.scan_indexes()
            self.scan_constraints()
            self.scan_sequences()
            
            return self.metadata
        finally:
            self.close()


class BaseConverter:
    """Base class for database object converters."""
    
    def __init__(self, metadata: Dict[str, Dict[str, Any]], output_dir: str):
        """
        Initialize the converter.
        
        Args:
            metadata: Database metadata from SybaseScanner
            output_dir: Directory to write output files
        """
        self.metadata = metadata
        self.output_dir = output_dir
        self.ddl_script = ""
        self.schema_script = ""
        self.data_script = ""
        
        # Create output directories
        self.create_output_dirs()
    
    def create_output_dirs(self) -> None:
        """Create output directory structure if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'ddl'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
    
    def convert(self) -> None:
        """Convert all database objects."""
        pass
    
    def write_ddl(self, filename: str, content: str) -> None:
        """Write DDL script to file."""
        with open(os.path.join(self.output_dir, 'ddl', filename), 'w') as f:
            f.write(content)
    
    def write_data_script(self, filename: str, content: str) -> None:
        """Write data loading script to file."""
        with open(os.path.join(self.output_dir, 'data', filename), 'w') as f:
            f.write(content)
    
    def convert_tables(self) -> str:
        """Convert tables to target database format."""
        pass
    
    def convert_views(self) -> str:
        """Convert views to target database format."""
        pass
    
    def convert_procedures(self) -> str:
        """Convert stored procedures to target database format."""
        pass
    
    def convert_functions(self) -> str:
        """Convert functions to target database format."""
        pass
    
    def convert_triggers(self) -> str:
        """Convert triggers to target database format."""
        pass
    
    def convert_indexes(self) -> str:
        """Convert indexes to target database format."""
        pass
    
    def convert_constraints(self) -> str:
        """Convert constraints to target database format."""
        pass
    
    def convert_sequences(self) -> str:
        """Convert sequences to target database format."""
        pass
    
    def create_data_load_scripts(self, batch_size: int = 1000, include_data: bool = False) -> None:
        """Create data load scripts."""
        pass


class SQLServerConverter(BaseConverter):
    """Converts Sybase objects to SQL Server format."""
    
    def convert(self) -> None:
        """Convert all database objects to SQL Server format."""
        logger.info("Converting to SQL Server format...")
        
        # Create main DDL script with proper order
        self.schema_script = "-- SQL Server Schema Creation Script\n"
        self.schema_script += f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Create schema creation statements
        schemas = set()
        for obj_type in self.metadata.values():
            for obj_meta in obj_type.values():
                if 'owner' in obj_meta and obj_meta['owner'] != 'dbo':
                    schemas.add(obj_meta['owner'])
        
        for schema in schemas:
            self.schema_script += f"IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = '{schema}')\n"
            self.schema_script += f"    EXEC('CREATE SCHEMA [{schema}]')\nGO\n\n"
        
        # Convert database objects
        tables_ddl = self.convert_tables()
        self.write_ddl('01_tables.sql', tables_ddl)
        self.schema_script += tables_ddl + "\n\n"
        
        indexes_ddl = self.convert_indexes()
        self.write_ddl('02_indexes.sql', indexes_ddl)
        self.schema_script += indexes_ddl + "\n\n"
        
        constraints_ddl = self.convert_constraints()
        self.write_ddl('03_constraints.sql', constraints_ddl)
        self.schema_script += constraints_ddl + "\n\n"
        
        views_ddl = self.convert_views()
        self.write_ddl('04_views.sql', views_ddl)
        self.schema_script += views_ddl + "\n\n"
        
        sequences_ddl = self.convert_sequences()
        self.write_ddl('05_sequences.sql', sequences_ddl)
        self.schema_script += sequences_ddl + "\n\n"
        
        procedures_ddl = self.convert_procedures()
        self.write_ddl('06_procedures.sql', procedures_ddl)
        self.schema_script += procedures_ddl + "\n\