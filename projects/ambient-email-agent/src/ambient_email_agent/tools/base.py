from langchain.tools import tool
from datetime import datetime
from pydantic import BaseModel


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """
    Write and send an email to a specified recipient.

    Use this tool when the user explicitly asks to send, write, or email
    a message to someone, and the recipient email address and message
    content can be extracted from the user input.

    Args:
        to (str): The recipient's email address.
        subject (str): The subject line of the email.
        content (str): The full body content of the email.

    Returns:
        str: A confirmation message indicating the email was sent,
        including recipient and subject details.

    Examples:
        User: "Send an email to john@example.com saying the meeting is postponed"
        → write_email(
            to="john@example.com",
            subject="Meeting Update",
            content="The meeting has been postponed."
        )
    """
    return f"Email sent to {to} with subject '{subject}' and content: {content}"


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: datetime,
    start_time: int,
) -> str:
    """
    Schedule a calendar meeting with specified participants and timing details.

    Use this tool when the user asks to schedule, arrange, or set up a meeting
    and provides (or implies) attendees, date, time, and duration.

    Args:
        attendees (list[str]): List of attendee email addresses or names.
        subject (str): Title or subject of the meeting.
        duration_minutes (int): Length of the meeting in minutes.
        preferred_day (datetime): Preferred calendar date for the meeting.
        start_time (int): Meeting start time in 24-hour format (e.g., 1330 for 1:30 PM).

    Returns:
        str: A confirmation message describing the scheduled meeting,
        including date, time, duration, and attendee count.

    Examples:
        User: "Schedule a 30-minute meeting with Alice and Bob on Friday at 2 PM
               to discuss the roadmap."
        → schedule_meeting(
            attendees=["Alice", "Bob"],
            subject="Roadmap Discussion",
            duration_minutes=30,
            preferred_day=datetime(2025, 1, 10),
            start_time=1400
        )
    """
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return (
        f"Meeting '{subject}' scheduled on {date_str} at {start_time} "
        f"for {duration_minutes} minutes with {len(attendees)} attendees"
    )

@tool
def check_calendar_availability(day: str) -> str:
    """
    Check available meeting time slots for a specified day.

    Use this tool when the user asks about availability, free time,
    or open meeting slots on a particular date.

    Args:
        day (str): The day to check availability for. This can be a
            natural language date (e.g., "Monday", "2025-01-10",
            "next Friday").

    Returns:
        str: A human-readable list of available time slots for the
        specified day.

    Examples:
        User: "Are you free this Friday?"
        → check_calendar_availability(day="Friday")

        User: "What time slots are open on 2025-01-10?"
        → check_calendar_availability(day="2025-01-10")
    """
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


@tool
class Question(BaseModel):
    """Ask the user a clarification question when required to proceed.

    This tool should be used when the assistant cannot confidently draft an
    email response due to missing, ambiguous, or conflicting information.
    The question should be concise, specific, and directly aimed at unblocking
    the next step in composing the response.

    Use this tool only when clarification from the user is strictly necessary.
    Do not ask follow-up questions if a reasonable assumption can be made.

    Attributes:
        content (str): The clarification question to ask the user.

    Usage Guidelines:
        - Ask only one clear question at a time.
        - Avoid conversational filler or explanations.
        - Do not include the drafted email in the question.
        - Do not explain why the question is being asked.
    """
    content: str



