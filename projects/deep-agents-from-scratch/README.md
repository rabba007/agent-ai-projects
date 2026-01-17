
# 🧱 Building Advanced AI Agents: A Hands-On Guide

The landscape of AI agents has evolved dramatically. We've moved beyond basic chatbots to sophisticated systems capable of handling complex, multi-step tasks. Projects like [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) demonstrate this shift, with typical workflows involving around 50 tool interactions. Similarly, tools like Claude Code are proving their versatility across diverse problem domains beyond their original scope.

After analyzing [successful agent architectures](https://docs.google.com/presentation/d/16aaXLu40GugY-kOpqDU4e-S0hD1FmHcNyF0rRRnb1OU/edit?slide=id.p#slide=id.p), we've identified three fundamental patterns that power these "deep" agents:

- **Structured planning workflows** with task recitation and tracking
- **State offloading** using virtual filesystem patterns  
- **Workload distribution** through specialized sub-agent architectures

This tutorial series demonstrates how to implement these architectural patterns using LangGraph, building everything from first principles.

## 🚀 Getting Started

### System Requirements

You'll need **Python 3.11+** to ensure full compatibility with LangGraph:

```bash
python3 --version
```

We also recommend installing [uv](https://docs.astral.sh/uv/), a modern Python package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/Users/$USER/.local/bin:$PATH"
```

### Setup Instructions

**Step 1:** Grab the repository:

```bash
git clone https://github.com/rabba007/agent-ai-projects.git
cd projects/deep_agents_from_scratch
```

**Step 2:** Install dependencies (uv handles virtual environment creation automatically):

```bash
uv sync
```

**Step 3:** Configure your environment variables:

```bash
touch .env
```

Add these keys to your `.env` file:

```bash
# Web search capability (required)
TAVILY_API_KEY=your_tavily_api_key_here

# Language model access (required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Observability tools (optional)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

**Step 4:** Launch the notebooks:

```bash
# Direct execution
uv run jupyter notebook

# Or activate the environment first
source .venv/bin/activate  # Windows: .venv\Scripts\activate
jupyter notebook
```

## 📚 Learning Path

This curriculum consists of five interconnected notebooks, each building upon previous concepts:

### Module 0: Agent Fundamentals

**File:** `0_create_agent.ipynb`

Introduction to the `create_agent` component, which provides:
- A ReAct (Reasoning-Action) loop serving as the architectural backbone
- Quick configuration and deployment
- Foundation for more advanced patterns

### Module 1: Planning Architecture  

**File:** `1_todo.ipynb`

Implementing intelligent task decomposition with TODO systems:
- Dynamic status tracking (pending → in_progress → completed)
- Contextual workflow management
- The `write_todos()` tool for breaking down complex objectives
- Strategies to maintain agent focus and avoid scope creep

### Module 2: Memory Systems

**File:** `2_files.ipynb`

Building virtual filesystems for persistent context management:
- Core operations: `ls()`, `read_file()`, `write_file()`, `edit_file()`
- Information persistence across conversation boundaries
- Simulating agent "memory" capabilities
- Token optimization through strategic data storage

### Module 3: Distributed Processing

**File:** `3_subagents.ipynb`

Designing sub-agent delegation patterns for complex workflows:
- Purpose-built agents with specialized toolsets
- Context boundaries preventing cross-talk and confusion
- Agent registry and the `task()` delegation interface
- Concurrent execution for independent workstreams

### Module 4: Production System

**File:** `4_full_agent.ipynb`

Synthesizing all patterns into a complete research agent:
- Unified TODO, filesystem, and sub-agent integration
- Live web search with smart context management
- Analytical reasoning and content synthesis tools
- End-to-end research workflows with LangGraph Studio support

Each module progressively adds sophistication, culminating in an enterprise-grade agent capable of real-world research and analytical tasks.
