def build_write_email_payload(tool_call, state, state_extractor):
    """Build interrupt payload for write_email tool"""
    
    # Extract email context if available
    email_context = ""
    if state_extractor:
        email_context = state_extractor(state)
    
    recipient = tool_call.get("args", {}).get("recipient", "unknown")
    
    return {
        "action_request": {
            "action": tool_call["name"],
            "args": tool_call.get("args", {})
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,   # User can provide feedback
            "allow_edit": True,      # User can edit the email
            "allow_accept": True,    # User can approve
        },
        "description": f"{email_context}\n📧 Email to {recipient} pending approval",
    }