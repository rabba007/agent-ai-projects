from typing import Callable, Any, Dict, Optional
from ambient_email_agent.custom_middleware.tool_interrupt_configuration import ToolInterruptConfig
from langchain.agents.middleware import AgentMiddleware
from langgraph.types import interrupt

class CustomInterruptMiddleware(AgentMiddleware):
    """
    Custom middleware handling interrupts for multiple tools.
    Converts agent tool calls to Agent Inbox interrupts with configurable payloads.
    """
    
    def __init__(
        self, 
        tool_configs: Dict[str, ToolInterruptConfig],
        tools_by_name: Dict[str, Any],
        state_extractor: Optional[Callable] = None,
    ):
        """
        Args:
            tool_configs: Dict mapping tool names to their interrupt configs
            tools_by_name: Dict mapping tool names to tool instances (for execution)
            state_extractor: Optional function to extract custom state (e.g., email_input)
        """
        self.tool_configs = tool_configs
        self.tools_by_name = tools_by_name
        self.state_extractor = state_extractor
        self.pending_decisions = {}  # Store decisions from interrupts
        
        # Tools that should NOT trigger interrupts (execute directly)
        self.direct_execute_tools = set(
            tool_name for tool_name, config in tool_configs.items() 
            if config is None
        )
    
    def after_model(self, state):
        """
        Trigger interrupts for configured tools.
        Runs after the model generates a response but before tools execute.
        """
        messages = state.get("messages", [])
        if not messages:
            return
        
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return
        
        # Collect results to update state
        result_messages = []
        
        # Process each tool call
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            
            # Skip if tool not in configs
            if tool_name not in self.tool_configs:
                continue
            
            config = self.tool_configs[tool_name]
            
            # If config is None, execute directly without interrupt
            if config is None:
                tool = self.tools_by_name[tool_name]
                observation = tool.invoke(tool_call.get("args", {}))
                result_messages.append({
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"]
                })
                continue
            
            # Build custom payload using tool-specific builder
            custom_payload = config.payload_builder(tool_call, state, self.state_extractor)
            
            # Trigger interrupt and get user response
            user_response = interrupt([custom_payload])
            
            # Handle the response and process accordingly
            response = user_response[0] if isinstance(user_response, list) else user_response
            
            # Process response using tool-specific processor
            processed_result = config.response_processor(
                response, 
                tool_call, 
                state,
                self.tools_by_name,
            )
            
            # Collect messages to update state
            result_messages.extend(processed_result.get("messages", []))
            
            # Store any flags (e.g., to go to END)
            self.pending_decisions[tool_call["id"]] = processed_result
        
        # Update state with messages if any were collected
        if result_messages:
            return {"messages": result_messages}