"""Perplexity Sonar Researcher Agent Implementation.

This module contains the logic for the Sonar Researcher agent, including the orchestrator
and tool nodes. It is extracted to avoid circular dependencies with the supervisor.
"""

from typing import List, Annotated
from typing_extensions import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage, filter_messages
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from deep_research_with_langgraph.state_scope import AgentState
from deep_research_with_langgraph.utils import think_tool, get_today_str
from deep_research_with_langgraph.prompts import (
    sonar_research_prompt, 
    compress_sonar_prompt,
    compress_research_human_message
)

# ===== CONFIGURATION =====

@tool
async def sonar_tool(query: str):
    """Query Perplexity Sonar model for detailed, cited answers to research questions.
    
    Args:
        query: The search query to send to the Sonar model.
        
    Returns:
        A formatted string containing the answer and a list of extracted sources/citations.
    """
    # Initialize the Sonar model
    # Note: Using 'sonar-pro' as a placeholder, can be configured via env vars or args if needed.
    # We use temperature=0 for consistent, factual results.
    model = init_chat_model("sonar", model_provider="perplexity", temperature=0)
    
    # helper for cleaner execution
    response = await model.ainvoke([HumanMessage(content=query)])
    
    # Extract citations
    # Perplexity API returns 'citations' in additional_kwargs as a list of URLs
    citations = response.additional_kwargs.get("citations", [])
    
    formatted_content = response.content
    
    # Append citations if they exist and are not already in the text (though usually they aren't explicit)
    if citations:
        formatted_content += "\n\n### Extracted Sources:\n"
        for i, url in enumerate(citations, 1):
            formatted_content += f"[{i}] {url}\n"
            
    return formatted_content

# Tools available to the orchestrator
search_tools = [sonar_tool, think_tool]

# Initialize the orchestrator model (The "reasoning" brain driving the loop)
# We use a capable model like GPT-4o for the orchestration logic
orchestrator_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
orchestrator_model_with_tools = orchestrator_model.bind_tools(search_tools)

# Model for compression/summarization
compress_model = init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0, max_tokens=32000)


# ===== STATE DEFINITIONS =====

class SonarResearcherState(AgentState):
    """State for the Sonar Researcher Agent.
    
    Extends AgentState to include the specific messages for the research loop.
    """
    researcher_messages: Annotated[List[BaseMessage], add_messages]
    compressed_research: str
    raw_notes: Annotated[List[str], lambda x, y: x + y]


# ===== NODES =====

async def orchestrator(state: SonarResearcherState):
    """The main decision-making node for the Sonar Researcher."""
    research_brief = state.get("research_brief", "")
    
    # Check if this is the first turn
    messages = state.get("researcher_messages", [])
    if not messages:
        # Initial instruction
        formatted_prompt = sonar_research_prompt.format(date=get_today_str())
        initial_messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"Research Brief: {research_brief}")
        ]
        
        # Invoke the orchestrator model
        response = await orchestrator_model_with_tools.ainvoke(initial_messages)
        
        # Return initial messages + response so they are added to state history
        return {"researcher_messages": initial_messages + [response]}
    
    # Invoke the orchestrator model with existing history
    response = await orchestrator_model_with_tools.ainvoke(messages)
    
    return {"researcher_messages": [response]}


async def tool_node(state: SonarResearcherState):
    """Execute tool calls requested by the orchestrator."""
    messages = state.get("researcher_messages", [])
    last_message = messages[-1]
    
    tool_calls = last_message.tool_calls
    results = []
    
    # Tools map
    tools_map = {t.name: t for t in search_tools}
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name in tools_map:
            result = await tools_map[tool_name].ainvoke(tool_args)
            
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            ))
            
    return {"researcher_messages": results}


def compress_sonar_results(state: SonarResearcherState):
    """Synthesize all Sonar findings into a clean notes format."""
    messages = state.get("researcher_messages", [])
    
    # Prepare the compression prompt
    system_prompt = compress_sonar_prompt.format(date=get_today_str())
    
    # Invoke compression model
    # We use the same rigorous human message as the standard researcher to ensure density and detail are preserved
    response = compress_model.invoke(
        [SystemMessage(content=system_prompt)] + messages + 
        [HumanMessage(content=compress_research_human_message.format(research_topic=state.get("research_brief", "research topic")))]
    )
    
    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            messages, 
            include_types=["tool", "ai"]
        )
    ]
    
    return {
        "compressed_research": response.content,
        "raw_notes": ["\n".join(raw_notes)]
    }


# ===== EDGES =====

def should_continue(state: SonarResearcherState) -> Literal["tool_node", "compress_sonar_results"]:
    """Decide whether to modify search or finish."""
    messages = state.get("researcher_messages", [])
    last_message = messages[-1]
    
    # If tool calls are present, execute them
    if last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, we assume the agent is done searching and ready to report
    return "compress_sonar_results"


# ===== GRAPH CONSTRUCTION =====

# 1. Build the Researcher Subgraph
researcher_builder = StateGraph(SonarResearcherState)
researcher_builder.add_node("orchestrator", orchestrator)
researcher_builder.add_node("tool_node", tool_node)
researcher_builder.add_node("compress_sonar_results", compress_sonar_results)

researcher_builder.add_edge(START, "orchestrator")
researcher_builder.add_conditional_edges(
    "orchestrator",
    should_continue
)
researcher_builder.add_edge("tool_node", "orchestrator")
# 'compress_sonar_results' is the end of this subgraph, but needs to output to 'notes' which is handled in the node
researcher_builder.add_edge("compress_sonar_results", END)

sonar_researcher = researcher_builder.compile()

