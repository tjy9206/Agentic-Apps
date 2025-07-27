# Agentic Apps

Welcome to the Agentic Apps repository! This is a collection of experimental AI agent systems built using the [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/). The purpose of this repository is to explore different patterns and capabilities of multi-agent systems for solving complex tasks.

## 🤖 About the Google Agent Development Kit (ADK)

The Google Agent Development Kit (ADK) is a framework designed to empower developers to build, manage, evaluate, and deploy sophisticated AI-powered agents. It provides a robust environment for creating both conversational and non-conversational agents capable of handling complex workflows by coordinating multiple specialized agents.

## 🛠️ General Setup and Installation

All agents in this repository are built with the ADK. To run any of them, you first need to set up your environment.

### 1. Clone the Repository

```bash
git clone [https://github.com/tjy9206/Agentic-Apps.git](https://github.com/tjy9206/Agentic-Apps.git)
cd Agentic-Apps
```

### 2. Set Up a Python Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies for this project.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install the Google Agent Development Kit (ADK)

Install the core ADK library using pip.

```bash
pip install google-adk
```

### 4. Set Your Google API Key

Nearly all agents will require a Google API Key to use Gemini models.

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Set it as an environment variable in your terminal. This single key will be used by all agents in this repository.

```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## ▶️ How to Run the Agents

The ADK comes with a powerful web-based development UI that is the primary way to interact with the agents in this repository.

### Project Structure

This repository is structured as a parent directory containing multiple, independent agent projects. For the `adk web` command to discover the agents, you must run it from the root of this repository.

```
Agentic-Apps/      <-- Run 'adk web' from here
├── stock_picker/
│   ├── __init__.py
│   └── agent.py
└── another_agent/
    ├── __init__.py
    └── agent.py
```

### Launching the Dev UI

From the root `Agentic-Apps` directory, run the following command:

```bash
adk web
```

This will start a local server, typically at `http://localhost:8000`. Open this URL in your browser.

- **Select an Agent**: Use the dropdown menu in the top-left corner to choose which agent system you want to interact with (e.g., `stock_picker`).
- **Interact**: Use the chat interface to send queries to the selected agent.
- **Inspect**: Use the `Events` and `Trace` tabs to inspect the agent's internal thoughts, tool calls, and execution flow.

## 📂 Available Agentic Apps

* [**stock_picker**](./stock_picker/README.md): A multi-agent system for long-term stock analysis.

*(More apps to come!)*
