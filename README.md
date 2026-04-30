# Automated Contract Risk Analysis

An intelligent, multi-agent AI system that automatically analyzes contracts for **freelance tech professionals**, identifies legal risks across critical clause categories, and generates a comprehensive risk report — powered by [CrewAI](https://crewai.com), [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/), and [Pinecone](https://www.pinecone.io/).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agent Pipeline](#agent-pipeline)
- [Clause Categories Analyzed](#clause-categories-analyzed)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Knowledge Base Setup](#knowledge-base-setup)
- [Running the Project](#running-the-project)
- [Output](#output)
- [Support](#support)

---

## Overview

Freelance tech professionals routinely sign contracts containing complex legal clauses — many of which carry significant risk. This project automates the contract review process by deploying a team of specialized AI agents that:

1. **Extract** structured clause data from a raw PDF contract
2. **Analyze** each clause against known risk patterns
3. **Research** similar legal precedents from a curated knowledge base
4. **Generate** a final, detailed risk report in Markdown format

The knowledge base is built from the **CUAD (Contract Understanding Atticus Dataset)**, a benchmark dataset of expert-annotated legal contracts, filtered to focus on the 10 clause types most relevant to CS freelancers.

---

## System Architecture

```
PDF Contract
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  CrewAI Sequential Pipeline          │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  Extractor   │───▶│  Analyst     │               │
│  │  Agent       │    │  Agent       │               │
│  │ (LlamaParse) │    │              │               │
│  └──────────────┘    └──────┬───────┘               │
│                             │                       │
│  ┌──────────────┐    ┌──────▼───────┐               │
│  │  Critic /    │◀───│  Researcher  │               │
│  │  Explainer   │    │  Agent       │               │
│  │  Agent       │    │ (Pinecone)   │               │
│  └──────┬───────┘    └──────────────┘               │
│         │                                           │
└─────────┼───────────────────────────────────────────┘
          │
          ▼
 output/final_contract_risk_report.md
```

The pipeline runs **sequentially**: each agent's output is passed as context to the next, ensuring outputs are grounded and progressively enriched.

---

## Agent Pipeline

| # | Agent | Tool | Responsibility |
|---|-------|------|----------------|
| 1 | **Extractor Agent** | `LlamaParseTool` | Parses the contract PDF and extracts raw clause text into structured data |
| 2 | **Analyst Agent** | — | Reviews extracted clauses and identifies risk indicators, ambiguities, and red flags |
| 3 | **Researcher Agent** | `PineconeSearchTool` | Searches the vector knowledge base for similar precedent clauses from the CUAD dataset |
| 4 | **Critic / Explainer Agent** | — | Synthesizes all findings into a professional risk report with plain-language explanations |

All agent behaviors, goals, and backstories are defined in `src/legal_team/config/agents.yaml`. All task descriptions and expected outputs are defined in `src/legal_team/config/tasks.yaml`.

---

## Clause Categories Analyzed

The system focuses on the 10 contract clause types most critical for CS freelance engagements:

| Clause Type | Why It Matters |
|---|---|
| **IP Ownership Assignment** | Determines who owns the code and deliverables you produce |
| **Non-Compete** | May restrict your ability to work with other clients |
| **Exclusivity** | Could prevent you from taking on other projects |
| **No-Solicit of Customers** | Restricts your ability to engage with the client's customers independently |
| **Cap on Liability** | Limits financial exposure in case of disputes or damages |
| **License Grant** | Defines what rights the client receives over your work |
| **Source Code Escrow** | Governs conditions under which source code is released to third parties |
| **Minimum Commitment** | Specifies guaranteed work volume or payment thresholds |
| **Non-Transferable License** | Prevents the client from selling or transferring your work to others |
| **Third Party Beneficiary** | Identifies if external parties can enforce the contract against you |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Multi-Agent Orchestration** | [CrewAI](https://crewai.com) v1.14.2 |
| **Document Parsing** | [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) |
| **Vector Database** | [Pinecone](https://www.pinecone.io/) (Serverless, AWS us-east-1) |
| **Embeddings** | `llama-text-embed-v2` via Pinecone Inference API (1024 dimensions, cosine) |
| **LLM Inference** | [Cerebras](https://cerebras.ai/) (`llama3.1-8b`) via Groq API |
| **Knowledge Dataset** | [CUAD v1](https://www.atticusprojectai.org/cuad) — Contract Understanding Atticus Dataset |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **Language** | Python >=3.10, <3.14 |

---

## Project Structure

```
Automated-Contract-Risk-Analysis/
├── preprocessing.ipynb                  # Data pipeline: CUAD → Pinecone knowledge base
└── legal_team/
    ├── pyproject.toml                   # Project metadata and dependencies
    ├── README.md
    ├── knowledge/
    │   └── user_preference.txt          # User context for agent personalization
    └── src/
        └── legal_team/
            ├── __init__.py
            ├── crew.py                  # Agent and task definitions (CrewAI @CrewBase)
            ├── main.py                  # Entry point — configures inputs and runs the crew
            └── config/
            │   ├── agents.yaml          # Agent roles, goals, and backstories
            │   └── tasks.yaml           # Task descriptions and expected outputs
            └── tools/
                ├── __init__.py
                └── custom_tool.py       # LlamaParseTool and PineconeSearchTool definitions
```

---

## Prerequisites

- Python `>=3.10, <3.14`
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for the following services:
  - **Groq** — LLM inference ([console.groq.com](https://console.groq.com))
  - **Pinecone** — Vector database ([app.pinecone.io](https://app.pinecone.io))
  - **LlamaParse** — PDF parsing ([cloud.llamaindex.ai](https://cloud.llamaindex.ai))

---

## Installation

**1. Install uv** (if not already installed):

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone the repository and navigate to the project:**

```bash
cd Automated-Contract-Risk-Analysis/legal_team
```

**3. Install project dependencies:**

```bash
crewai install
```

This command uses `uv` to resolve and install all dependencies listed in `pyproject.toml`.

---

## Configuration

**1. Create a `.env` file** in `legal_team/` with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
LLAMAPARSE_API_KEY=your_llamaparse_api_key_here
```

**2. Set the contract file path** in `src/legal_team/main.py`:

```python
inputs = {
    'file_path': 'path/to/your/contract.pdf'
}
```

**3. (Optional) Customize agent behavior** by editing:
- `src/legal_team/config/agents.yaml` — agent roles, goals, and backstories
- `src/legal_team/config/tasks.yaml` — task descriptions and expected output formats

---

## Knowledge Base Setup

Before running the analysis pipeline, you must populate the Pinecone vector database with legal precedent data from the CUAD dataset.

**1. Download the CUAD dataset:**

Download `CUAD_v1.zip` from the [CUAD project page](https://www.atticusprojectai.org/cuad) and extract it so that `CUAD_v1/master_clauses.csv` is accessible from the notebook directory.

**2. Run the preprocessing notebook:**

Open and run all cells in `preprocessing.ipynb`. This notebook will:
- Load and explore the CUAD master clauses CSV
- Filter for the 10 CS freelancer-relevant clause types
- Embed all filtered clauses using `llama-text-embed-v2` via the Pinecone Inference API
- Create a Pinecone Serverless index named `contracts` (1024 dimensions, cosine similarity)
- Upload all embedded clauses in batches of 50 with rate-limit protection

> **Note:** The notebook supports **resumable uploads** — if the process is interrupted, re-running the upload cell will automatically detect how many vectors are already saved and resume from where it left off.

---

## Running the Project

Once the knowledge base is populated and the `.env` is configured, run the full multi-agent analysis from the `legal_team/` directory:

```bash
uv run --active run_crew
```

The crew will sequentially execute all four agents and save the final report.

---

## Output

The final risk report is saved to:

```
legal_team/output/final_contract_risk_report.md
```

The report includes:
- A structured breakdown of each identified clause
- Risk level assessment per clause
- Plain-language explanations of potential legal implications
- Comparisons against similar precedent clauses from the CUAD knowledge base
- Recommendations for negotiation or review

---

## Support

- CrewAI Documentation: [docs.crewai.com](https://docs.crewai.com)
- CrewAI GitHub: [github.com/joaomdmoura/crewai](https://github.com/joaomdmoura/crewai)
- CrewAI Discord: [discord.com/invite/X4JWnZnxPb](https://discord.com/invite/X4JWnZnxPb)
- Pinecone Docs: [docs.pinecone.io](https://docs.pinecone.io)
- LlamaParse Docs: [docs.llamaindex.ai](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/)
