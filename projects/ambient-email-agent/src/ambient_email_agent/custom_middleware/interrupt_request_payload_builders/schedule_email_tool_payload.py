def build_schedule_meeting_payload(tool_call, state, state_extractor):
    """Build interrupt payload for schedule_meeting tool"""
    
    meeting_title = tool_call.get("args", {}).get("title", "Meeting")
    attendees = tool_call.get("args", {}).get("attendees", [])
    
    return {
        "action_request": {
            "action": tool_call["name"],
            "args": tool_call.get("args", {})
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": f"📅 Scheduling '{meeting_title}' with {len(attendees)} attendees pending approval",
    }

__all__ = [
    "build_schedule_meeting_payload",
]