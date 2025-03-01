# Investor Relations Agent System

An agentic application for investor relations analysis using OpenAI's API and a multi-agent architecture.

## Overview

This system uses a collection of specialized AI agents to analyze company information for investor relations purposes. The agents work together to:

1. Process user input to extract company names
2. Verify company listing status
3. Collect relevant financial data
4. Identify peer companies for comparison
5. Analyze collected data for insights
6. Generate insightful questions an investor might ask
7. Formulate comprehensive answers to those questions

## Project Structure

- `investor_relations_agent_system.py` - Main system implementation with all agents
- `cli.py` - Command-line interface for interacting with the system
- `requirements.txt` - Required Python dependencies
- `.env.template` - Template for environment variables
- `output/` - Directory where analysis results are saved

## Setup Instructions

### Prerequisites

- Python 3.13 or higher
- PyCharm IDE (recommended)
- OpenAI API key

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd investor-relations-agent-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```
   cp .env.template .env
   ```
   Then edit `.env` and add your OpenAI API key.

### Running the Application

Run the application using the CLI interface:

```
python cli.py
```

The CLI will prompt you to enter a company name or query. The system will then process your query through all agents and display the results.

## Agent Architecture

The system is built around a hierarchy of specialized agents:

### Top-level Agents

1. **Input Processing Agent** - Processes user input to extract company names
2. **Company Verification Agent** - Verifies company existence and listing status
3. **Data Collection Agent** - Collects relevant company data
4. **Peer Identification Agent** - Identifies similar companies for comparison
5. **Analysis Agent** - Analyzes collected data for insights
6. **Question Generation Agent** - Generates insightful investor questions
7. **Answer Formulation Agent** - Formulates comprehensive answers

### Sub-agents

Each top-level agent has multiple specialized sub-agents that handle specific aspects of the task. For example:

- The **Input Processing Agent** uses sub-agents for name normalization, input validation, and disambiguation.
- The **Analysis Agent** uses sub-agents for topic extraction, financial metric analysis, sentiment analysis, etc.

## Customization

You can customize the system by:

1. Modifying the agent prompts in each agent class
2. Adding new sub-agents for additional functionality
3. Adjusting the model selection for different tasks (DEFAULT_MODEL and FAST_MODEL)
4. Extending the data collection to use real APIs instead of simulated data

## Output

Analysis results are saved as JSON files in the `output/` directory with the following structure:

```json
{
  "status": "success",
  "company_name": "Company Name",
  "company_info": { ... },
  "analysis": { ... },
  "questions_and_answers": [
    {
      "question": "Question text",
      "answer": "Answer text",
      "category": "strategic|financial|risk|competitive",
      "rank": 1
    },
    ...
  ]
}
```

## License

[Your license information here]
