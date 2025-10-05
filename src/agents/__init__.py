from typing import Dict, Type, List, Optional
from pathlib import Path
import importlib
import inspect
import pkgutil
import logging

from .base_agent import BaseAgent
from .pdf_agent import PDFAgent

class AgentRegistry:
    """Manages file type agents and routes files to the appropriate handlers."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.extension_map: Dict[str, BaseAgent] = {}
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new file agent."""
        if not isinstance(agent, BaseAgent):
            raise ValueError("Only subclasses of BaseAgent can be registered")
            
        agent_name = agent.__class__.__name__
        self.agents[agent_name] = agent
        
        # Update extension mapping
        for ext in agent.supported_formats:
            ext = ext.lower()
            if ext in self.extension_map:
                logging.warning(f"Extension {ext} is already handled by {self.extension_map[ext].__class__.__name__}")
            self.extension_map[ext] = agent
            
    def get_agent_for_file(self, file_path: str) -> Optional[BaseAgent]:
        """Get the appropriate agent for the given file."""
        ext = Path(file_path).suffix.lower()
        return self.extension_map.get(ext)
    
    def discover_agents(self, package_name: Optional[str] = None) -> None:
        """Auto-discover and register all agents in the specified package.
        
        Args:
            package_name: The package to search for agents. Defaults to this package.
        """
        package_name = package_name or __name__
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logging.warning(f"Could not import package {package_name}")
            return
            
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseAgent) and 
                        obj != BaseAgent and 
                        not inspect.isabstract(obj)):
                        agent = obj()
                        self.register_agent(agent)
                        logging.info(f"Registered agent: {agent.__class__.__name__}")
            except ImportError as e:
                logging.warning(f"Failed to import {full_module_name}: {e}")

# Global registry instance
registry = AgentRegistry()

# Register built-in agents
registry.register_agent(PDFAgent())

# Discover additional agents from the agents directory
registry.discover_agents()
