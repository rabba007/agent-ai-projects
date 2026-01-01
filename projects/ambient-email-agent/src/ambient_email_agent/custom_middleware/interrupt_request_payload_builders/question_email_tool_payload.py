def build_question_payload(tool_call, state, state_extractor):
    """Build interrupt payload for Question tool"""
    
    question_text = tool_call.get("args", {}).get("question", "Question")
    
    return {
        "action_request": {
            "action": tool_call["name"],
            "args": tool_call.get("args", {})
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,    # User must provide answer
            "allow_edit": False,      # Cannot edit question
            "allow_accept": False,    # No approval needed
        },
        "description": f"❓ {question_text}",
    }