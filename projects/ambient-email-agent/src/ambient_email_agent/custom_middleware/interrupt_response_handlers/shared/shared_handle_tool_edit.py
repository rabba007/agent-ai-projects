# Now use this helper in your handlers


def handle_tool_edit(tool_call, edited_args, state, tools_by_name, tool_name):
    """
    Handles the edit case for any tool.
    
    Returns a list of messages:
    1. Updated AI message with new tool call arguments
    2. Tool execution result
    """
    result_messages = []
    tool_call_id = tool_call["id"]
    
    # Get the most recent AI message from state
    ai_message = state["messages"][-1]
    
    # Create a new list of tool calls, replacing the one being edited
    # This ensures immutability - we don't modify the original list
    updated_tool_calls = [
        tc for tc in ai_message.tool_calls 
        if tc.get("id") != tool_call_id
    ] + [
        {
            "type": "tool_call",
            "name": tool_call["name"],
            "args": edited_args,
            "id": tool_call_id
        }
    ]
    
    # Create a new copy of the message with updated tool calls
    # The add_messages reducer will merge this by ID, effectively updating the message
    updated_ai_message = ai_message.model_copy(
        update={"tool_calls": updated_tool_calls}
    )
    
    # Add the updated AI message to results
    result_messages.append(updated_ai_message)
    
    # Execute the tool with edited args
    tool = tools_by_name[tool_name]
    observation = tool.invoke(edited_args)
    
    # Add the tool response
    result_messages.append({
        "role": "tool",
        "content": observation,
        "tool_call_id": tool_call_id
    })
    
    return result_messages


__all__ = ["handle_tool_edit"]