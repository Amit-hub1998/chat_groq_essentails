# Hugging Face Researcher with Groq + MCP

This project demonstrates a Groq-powered research agent that uses the OpenAI Python SDK against Groq's OpenAI-compatible endpoint and connects to a Hugging Face MCP server to discover and call tools during a single model run.

The notebook uses a reusable skill prompt to instruct the model to produce a concise research summary with a verdict, markdown table, and recommendation section.

## Overview

The workflow is:

1. Define a system prompt / skill
2. Pass a user query
3. Attach an MCP tool server
4. Call `client.responses.create(...)` once
5. Let Groq handle the reasoning + tool loop server-side
6. Read the final answer from `response.output_text`

This avoids manually coding a `while` loop to check for tool calls and finish states yourself.

## Why this project matters

The key idea is server-side orchestration:

- You send one request with the prompt and query
- The model reasons about whether it needs tools
- It discovers tools from the MCP server
- It calls the tool when needed
- Results come back to the model
- The model continues until it has enough information to produce the final answer

The notebook includes a trace helper that shows the agent steps while it works.

## Repository contents

- `managed_agent.ipynb` — main notebook with the workflow
- `hf_researcher_responses_mcp_flow.svg` — diagram explaining the flow
- `README.md` — GitHub-facing documentation

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install openai python-dotenv
```

### 3. Set environment variables

Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_api_key_here
```

## Notebook usage

Open `managed_agent.ipynb` in VS Code or Jupyter and run the cells in order.

### Core setup cell

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-120b"
```

### Skill prompt used by the model

```python
RESEARCHER_SKILL = """You are a Hugging Face model research specialist.
When reporting on models, always:
- Lead with a one-line verdict naming the best pick and why
- Present findings as a markdown table (Model | Task | Downloads | Likes | Score)
- End with a short 'Recommendation' paragraph
- Only use data returned by the tools
"""
```

### MCP tool definition

```python
TOOLS = [
    {
        "type": "mcp",
        "server_label": "Huggingface",
        "server_url": "https://huggingface.co/mcp",
        "server_description": "Search and access AI models, datasets, and spaces from Hugging Face Hub",
        "require_approval": "never",
    }
]
```

### Main request

```python
def research_models(query: str):
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": RESEARCHER_SKILL},
            {"role": "user", "content": f"Query: {query}"},
        ],
        tools=TOOLS,
    )
    return response
```

Then call it with a task like:

```python
response = research_models(
    "Find the top 5 trending text-generation models this month on Hugging Face."
)

print(response.output_text)
```

## Example result format

The model is instructed to provide output in this structure:

- one-line verdict
- markdown table with model details
- recommendation paragraph

Example shape:

```markdown
Best pick: XLM-R for multilingual generation because it is highly ranked and widely used.

| Model | Task | Downloads | Likes | Score |
| --- | --- | ---: | ---: | ---: |
| ... | ... | ... | ... | ... |

Recommendation: Choose ... for ...
```

## Troubleshooting

### Missing `responses` attribute

If you get an error like `AttributeError: 'OpenAI' object has no attribute 'responses'`, upgrade the SDK:

```bash
pip install --upgrade openai
```

Then restart the notebook kernel.

### Missing environment variable

If `GROQ_API_KEY` is missing, verify the `.env` file exists and contains the correct key.

### Tool call issues

If the model cannot reach the Hugging Face MCP server, confirm:

- the URL is `https://huggingface.co/mcp`
- the network is available
- the model supports the tool-calling pattern

## Notes

- This project depends on the Groq-compatible Responses API.
- The notebook intentionally demonstrates a lightweight agent pattern without a hand-written tool loop.
- The trace helper is useful for debugging reasoning, tool discovery, and tool invocation behavior.

## License

This project is intended for educational and experimentation purposes.
