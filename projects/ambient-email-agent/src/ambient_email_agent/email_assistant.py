from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)

from ambient_email_agent.tools.base import write_email,schedule_meeting,check_calendar_availability
from ambient_email_agent.schemas import State, RouterSchema
from langgraph.types import Command
from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from ambient_email_agent.utils import parse_email, format_email_markdown
from ambient_email_agent.prompts import AGENT_TOOLS_PROMPT, agent_system_prompt, default_cal_preferences, default_response_preferences, triage_system_prompt, default_background, default_triage_instructions, triage_user_prompt
from langchain.agents import create_agent


# Initialize the LLM for use with router / structured output
model_gemini_flash = init_chat_model("gemini-2.5-flash", model_provider="google_genai", timeout=30, temperature=0)
model_llama_groq = init_chat_model("llama-3.1-8b-instant", model_provider="groq", timeout=30, temperature=0)
router_llm = model_llama_groq.with_structured_output(RouterSchema)


#Node
def triage_router(state:State)->Command[Literal["response_agent", END]]:
    """Analyze email content to decide if we should respond, notify, or ignore."""

    author, to, subject, email_thread = parse_email(state["email_input"])

    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    result = router_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email: \n\n{format_email_markdown(subject, author, to, email_markdown)}",
                }
            ],
            "classification_decision": result.classification,
        }
        
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        goto = END
        update =  {
            "classification_decision": result.classification,
        }
        
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        # For now, we go to END. But we will add to this later!
        goto = END
        update = {
            "classification_decision": result.classification,
        }
        
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


system_prompt_for_email_writer_agent = agent_system_prompt.format(
                        tools_prompt=AGENT_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences,
                        cal_preferences=default_cal_preferences, 
                    )

email_writer_agent = create_agent(
    model=model_gemini_flash,
    tools=[write_email, schedule_meeting, check_calendar_availability],
    system_prompt=system_prompt_for_email_writer_agent,
    middleware=[ModelCallLimitMiddleware(
        thread_limit=20, run_limit=10, exit_behavior='error'
    )]
)

#Node
def call_email_writer_agent(state:State):
    response = email_writer_agent.invoke({"messages": [state.get('messages')[-1]]})
    return {"messages": [response["messages"][-1]]} 

overall_workflow = (StateGraph(State)
.add_node("triage_router", triage_router)
.add_node("response_agent",call_email_writer_agent)
.add_edge(START, "triage_router")
.add_edge("triage_router", END)
)

email_assistant = overall_workflow.compile()

    
    

