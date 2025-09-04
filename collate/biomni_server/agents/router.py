"""Query routing logic for selecting appropriate agents."""


class QueryRouter:
    """Intelligent routing between A1 and ReAct agents."""
    
    def __init__(self, default_agent: str, react_enabled: bool):
        self.default_agent = default_agent
        self.react_enabled = react_enabled
    
    def classify_query(self, query: str) -> str:
        """
        Classify query to determine which agent to use.
        
        Args:
            query: The user query to classify
            
        Returns:
            "a1" for code execution needs, "react" for tool-based queries
        """
        # If ReAct is disabled, always use A1
        if not self.react_enabled:
            return "a1"
        
        # TODO: Add more sophisticated query classification logic
        # For now, use configured default agent
        return self.default_agent
