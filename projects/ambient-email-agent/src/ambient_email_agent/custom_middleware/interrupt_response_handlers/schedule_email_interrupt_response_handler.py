from ambient_email_agent.custom_middleware.interrupt_response_handlers.shared.shared_handle_tool_edit import handle_tool_edit


def process_schedule_meeting_response(response, tool_call, state, tools_by_name):
    """Process user's decision for schedule_meeting tool"""
    
    decision_type = response.get("type")
    tool_call_id = tool_call["id"]
    result_messages = []
    
    if decision_type == "accept":
        tool = tools_by_name["schedule_meeting"]
        observation = tool.invoke(tool_call.get("args", {}))
        result_messages.append({
            "role": "tool",
            "content": observation,
            "tool_call_id": tool_call_id
        })
        
    elif decision_type == "reject":
        result_messages.append({
            "role": "tool",
            "content": "User ignored this calendar meeting draft. Ignore this meeting and end the workflow.",
            "tool_call_id": tool_call_id
        })
        
    elif decision_type == "edit":
        edited_args = response.get("args", {}).get("args", {})
        result_messages.extend(
            handle_tool_edit(tool_call, edited_args, state, tools_by_name, "schedule_meeting")
        )
        
    elif decision_type == "response":
        user_feedback = response.get("args", "")
        result_messages.append({
            "role": "tool",
            "content": f"User gave feedback, which we can incorporate into the meeting request. Feedback: {user_feedback}",
            "tool_call_id": tool_call_id
        })
    
    return {"messages": result_messages}


__all__ = ["process_schedule_meeting_response"]