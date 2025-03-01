# Sybase Database Converter

This tool converts Sybase stored procedures to other database formats, currently supporting:
- Microsoft SQL Server
- Oracle
- Java
- JavaScript

## Features

- Converts Sybase data types to their target database equivalent
- Updates SQL syntax to match target database requirements
- Handles function conversion (e.g., date functions, transactions, etc.)
- Preserves directory structure in output
- Adds comments to the converted code to explain changes
- Generates timestamped log files with details of the conversion process

## Usage

```bash
python sybase_converter.py --input <input_dir> --output <output_dir> --target <target_db>
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--input`, `-i` | Input directory containing Sybase stored procedures |
| `--output`, `-o` | Output directory for converted procedures |
| `--target`, `-t` | Target database type (mssql, oracle, java, javascript, default: mssql) |
| `--recursive`, `-r` | Process subdirectories recursively |
| `--verbose`, `-v` | Enable verbose logging |

### Examples

Convert Sybase procedures to MS SQL Server:
```bash
python sybase_converter.py --input ./sybase_procs --output ./converted
```

Convert Sybase procedures to Oracle:
```bash
python sybase_converter.py --input ./sybase_procs --output ./converted --target oracle
```

Convert Sybase procedures to Java classes:
```bash
python sybase_converter.py --input ./sybase_procs --output ./converted --target java
```

## Conversion Output

The tool creates a subdirectory for each target database format within the specified output directory. For example:
- `./converted/mssql/` - MS SQL Server converted procedures
- `./converted/oracle/` - Oracle converted procedures
- `./converted/java/` - Java converted classes
- `./converted/javascript/` - JavaScript converted files

Each converted file includes a header comment indicating:
- The original file name
- The conversion date and time
- The target database type

## Java and JavaScript Conversion Notes

The Java conversion transforms SQL stored procedures to Java classes that can utilize JDBC for database operations. Similarly, JavaScript conversion creates classes that can be used in a JavaScript environment.

Since these are completely different programming paradigms, the converted files should be considered as starting points that will require additional manual adjustments.

## General Notes

- The tool adds comments to the converted code to mark and explain changes
- Log files are automatically generated with timestamps
- Some complex constructs may require manual adjustment after conversion
