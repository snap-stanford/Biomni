"""Agent initialization and management."""

import sys
from typing import Optional
from biomni.agent.a1 import A1
from biomni.agent.react import react
from ..config import BiomniConfig


class AgentManager:
    """Manages initialization and lifecycle of Biomni agents."""
    
    def __init__(self, config: BiomniConfig):
        self.config = config
        self.a1_agent: Optional[A1] = None
        self.react_agent: Optional[react] = None
        
    def initialize_agents(self) -> None:
        """Initialize all configured agents."""
        self._initialize_a1_agent()
        if self.config.enable_react:
            self._initialize_react_agent()
        else:
            print("ReAct agent disabled via BIOMNI_ENABLE_REACT", file=sys.stderr)
    
    def _initialize_a1_agent(self) -> None:
        """Initialize the A1 agent."""
        print("Initializing A1 agent...", file=sys.stderr)
        self.a1_agent = A1(
            path=self.config.data_path,
            llm=self.config.llm_model,
            use_tool_retriever=True,
            timeout_seconds=self.config.a1_timeout,
        )
    
    def _initialize_react_agent(self) -> None:
        """Initialize the ReAct agent."""
        print("Initializing ReAct agent...", file=sys.stderr)
        self.react_agent = react(
            path=self.config.data_path,
            llm=self.config.llm_model,
            use_tool_retriever=True,
            timeout_seconds=self.config.react_timeout,
        )
        
        # Configure ReAct agent
        self.react_agent.configure(
            plan=True,
            reflect=True,
            data_lake=True,
            library_access=True,
        )
    
    def get_available_agents(self) -> list[str]:
        """Get list of available agent types."""
        agents = ["a1"]
        if self.react_agent is not None:
            agents.append("react")
        return agents
    
    def get_agent_by_type(self, agent_type: str):
        """Get agent instance by type."""
        if agent_type == "a1":
            return self.a1_agent
        elif agent_type == "react":
            return self.react_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
