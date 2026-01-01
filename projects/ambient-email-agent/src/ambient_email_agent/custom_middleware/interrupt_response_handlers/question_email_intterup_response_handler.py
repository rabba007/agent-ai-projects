def process_question_response(response, tool_call, state, tools_by_name):
    """Process user's response for Question tool"""
    decision_type = response.get("type")
    tool_call_id = tool_call["id"]
    result_messages = []
    
    
        
    if decision_type == "reject":
        # User ignored the question - end workflow
        result_messages.append({
            "role": "tool",
            "content": "User ignored this question. Ignore this email and end the workflow.",
            "tool_call_id": tool_call_id
        })
        
        
    elif decision_type == "response":
        # User answered the question - primary flow
        user_answer = response.get("args", "")
        result_messages.append({
            "role": "tool",
            "content": f"User answered the question, which we can use for any follow-up actions. Answer: {user_answer}",
            "tool_call_id": tool_call_id
        })
    
    return {"messages": result_messages}