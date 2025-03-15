"""
Compatibility module for handling different versions of dependencies
"""
import importlib.metadata
from typing import Dict, Any, Optional, Union, List

def get_crewai_version() -> str:
    """Get the installed version of crewai"""
    try:
        return importlib.metadata.version("crewai")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"

def create_task(
    description: str, 
    agent: Any, 
    depends_on: Optional[List[Any]] = None,
    **kwargs
) -> Any:
    """Create a Task with compatibility for different crewai versions"""
    from crewai import Task
    
    if depends_on is None:
        depends_on = []
    
    crewai_version = get_crewai_version()
    
    # For crewai >= 0.34.0
    if parse_version(crewai_version) >= parse_version("0.34.0"):
        # Make sure expected_output is provided for newer versions
        if "expected_output" not in kwargs:
            kwargs["expected_output"] = "Analysis and documentation of the code."
    
    # Create and return the task with all parameters
    return Task(
        description=description,
        agent=agent,
        depends_on=depends_on,
        **kwargs
    )

def parse_version(version: str) -> tuple:
    """Parse version string into comparable tuple"""
    return tuple(map(int, version.split(".")))
