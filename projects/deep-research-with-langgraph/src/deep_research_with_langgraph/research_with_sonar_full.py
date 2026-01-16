"""Perplexity Sonar Research Agent Workflow.

This module implements the standalone research workflow using Perplexity's Sonar models
via the shared supervisor agent.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from deep_research_with_langgraph.state_scope import AgentState, AgentInputState
from deep_research_with_langgraph.research_agent_scope import clarify_with_user, write_research_brief
from deep_research_with_langgraph.utils import get_today_str
from deep_research_with_langgraph.prompts import sonar_final_report_prompt
from deep_research_with_langgraph.multi_agent_supervisor import supervisor_agent

# Model for final report writing
writer_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0, max_tokens=12000)

async def final_report_generation(state: AgentState):
    """Generate the final report based on gathered notes."""
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)
    research_brief = state.get("research_brief", "")
    
    formatted_prompt = sonar_final_report_prompt.format(
        research_brief=research_brief,
        findings=findings,
        date=get_today_str()
    )
    
    response = await writer_model.ainvoke([HumanMessage(content=formatted_prompt)])
    
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"Here is the final report:\n\n{response.content}")]
    }

def set_sonar_mode(state: AgentState):
    """Set the research mode to sonar."""
    return {"research_mode": "sonar"}


# ===== GRAPH CONSTRUCTION =====

# Build the Main Workflow using Supervisor
sonar_research_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add nodes
sonar_research_builder.add_node("clarify_with_user", clarify_with_user)
sonar_research_builder.add_node("write_research_brief", write_research_brief)
sonar_research_builder.add_node("supervisor_subgraph", supervisor_agent)
sonar_research_builder.add_node("sonar_mode_setter", set_sonar_mode)
sonar_research_builder.add_node("final_report_generation", final_report_generation)

# Add edges
sonar_research_builder.add_edge(START, "clarify_with_user")
# We assume clarify logic handles its own routing (it returns Command)

sonar_research_builder.add_edge("write_research_brief", "sonar_mode_setter")
sonar_research_builder.add_edge("sonar_mode_setter", "supervisor_subgraph")
sonar_research_builder.add_edge("supervisor_subgraph", "final_report_generation")
sonar_research_builder.add_edge("final_report_generation", END)

# Compile
agent = sonar_research_builder.compile()
