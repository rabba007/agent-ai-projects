from ambient_email_agent.custom_middleware.interrupt_request_payload_builders.question_email_tool_payload import build_question_payload
from ambient_email_agent.custom_middleware.interrupt_request_payload_builders.schedule_email_tool_payload import build_schedule_meeting_payload
from ambient_email_agent.custom_middleware.interrupt_response_handlers.question_email_intterup_response_handler import process_question_response
from ambient_email_agent.custom_middleware.interrupt_response_handlers.schedule_email_interrupt_response_handler import process_schedule_meeting_response
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from ambient_email_agent.tools.base import write_email, schedule_meeting, check_calendar_availability, Question
from ambient_email_agent.schemas import RouterSchema, State
from ambient_email_agent.utils import extract_email_context, format_email_markdown, format_for_display, parse_email
from ambient_email_agent.prompts import HITL_TOOLS_PROMPT, default_cal_preferences, default_response_preferences, triage_system_prompt, triage_user_prompt, default_background,default_triage_instructions\
    , agent_system_prompt_hitl

from ambient_email_agent.custom_middleware.custom_interrupt_middleware import CustomInterruptMiddleware, ToolInterruptConfig
from ambient_email_agent.custom_middleware.interrupt_request_payload_builders.write_email_tool_payload import build_write_email_payload
from ambient_email_agent.custom_middleware.interrupt_response_handlers.write_tool_interrupt_response_handler import process_write_email_response



load_dotenv(override=True)

# Get tools
tools = [write_email, schedule_meeting,check_calendar_availability, Question]
tools_by_name = {tool.name: tool for tool in tools}


# Initialize the LLM for use with router / structured output
model_gemini_flash = init_chat_model("gemini-2.5-flash", model_provider="google_genai", timeout=30, temperature=0)
model_llama_groq = init_chat_model("llama-3.1-8b-instant", model_provider="groq", timeout=30, temperature=0)
router_llm = model_llama_groq.with_structured_output(RouterSchema)


# Nodes 
def triage_router(state: State) -> Command[Literal["triage_interrupt_handler", "response_agent", END]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """

    # Parse the email input
    author, to, subject, email_thread = parse_email(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    # Run the router LLM
    result = router_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    # Process the classification decision
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Next node
        goto = "response_agent"
        # Update the state
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")

        # Next node
        goto = END
        # Update the state
        update = {
            "classification_decision": classification,
        }

    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information") 

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "classification_decision": classification,
        }

    else:
        raise ValueError(f"Invalid classification: {classification}")
    return Command(goto=goto, update=update)


def triage_interrupt_handler(state: State) -> Command[Literal["response_agent", END]]:
    """Handles interrupts from the triage step"""
    
    # Parse the email input
    author, to, subject, email_thread = parse_email(state["email_input"])

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Create messages
    messages = [{"role": "user",
                "content": f"Email to notify user about: {email_markdown}"
                }]

    # Create interrupt for Agent Inbox
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": {
            "allow_ignore": True,  
            "allow_respond": True, 
            "allow_edit": False, 
            "allow_accept": False,  
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    # Agent Inbox responds with a list  
    response = interrupt([request])[0]

    # If user provides feedback, go to response agent and use feedback to respond to email   
    if response["type"] == "response":
        # Add feedback to messages 
        user_input = response["args"]
        # Used by the response agent
        messages.append({"role": "user",
                        "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"
                        })
        # Go to response agent
        goto = "response_agent"

    # If user ignores email, go to END
    elif response["type"] == "ignore":
        goto = END

    # Catch all other responses
    else:
        raise ValueError(f"Invalid response: {response}")

    # Update the state 
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)




tool_configs = {
    "write_email": ToolInterruptConfig(
        payload_builder=build_write_email_payload,
        response_processor=process_write_email_response,
        description="Email sending requires approval"
    ),
    "schedule_meeting": ToolInterruptConfig(
        payload_builder=build_schedule_meeting_payload,
        response_processor=process_schedule_meeting_response,
        description="Meeting scheduling requires approval"
    ),
    "Question": ToolInterruptConfig(
        payload_builder=build_question_payload,
        response_processor=process_question_response,
        description="Question requires user answer"
    ),
}

system_prompt_for_email_writer_agent=agent_system_prompt_hitl.format(tools_prompt=HITL_TOOLS_PROMPT, 
                                                                     background=default_background,
                                                                     response_preferences=default_response_preferences, 
                                                                     cal_preferences=default_cal_preferences)

response_agent = create_agent(
    model=model_gemini_flash,
    tools=tools,
    system_prompt=system_prompt_for_email_writer_agent,
    state_schema=State,
    middleware=[ModelCallLimitMiddleware(
        thread_limit=20, run_limit=10, exit_behavior='error'
    ), CustomInterruptMiddleware(
        tool_configs=tool_configs,
        tools_by_name=tools_by_name,
        state_extractor=extract_email_context,
    )]
)

#Node
def call_email_writer_agent(state:State):
    response = response_agent.invoke({"messages": [state.get('messages')[-1]]})
    return {"messages": [response["messages"][-1]]} 


# Build overall workflow
overall_workflow = (
    StateGraph(State)
    .add_node("triage_router",triage_router)
    .add_node("triage_interrupt_handler",triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    
)

email_assistant_hitl_v2 = overall_workflow.compile()


