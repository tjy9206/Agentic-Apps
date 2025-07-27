import os
import asyncio
import yfinance as yf
from typing import Dict

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, AgentTool, google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- 1. Define Specialist Tools ---------------------------------------------
# For this example, we'll use the yfinance library, which doesn't require an
# API key. In a production system, you would replace this with a more robust
# financial data provider and likely pass an API key from environment variables.

def get_financial_data(ticker: str) -> Dict:
    """
    Retrieves key financial data for a given stock ticker using yfinance.

    Args:
        ticker: The stock ticker symbol (e.g., 'GOOGL', 'MSFT').

    Returns:
        A dictionary containing key financial metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key metrics. yfinance keys can sometimes be missing, so we use .get()
        financials = {
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "price_to_book": info.get("priceToBook"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "average_volume": info.get("averageVolume"),
            "short_summary": info.get("longBusinessSummary"),
        }
        return financials
    except Exception as e:
        return {"error": f"Could not retrieve data for {ticker}. Error: {str(e)}"}

# Wrap the function in ADK's FunctionTool
financial_data_tool = FunctionTool(func=get_financial_data)


# --- 2. Define Specialist Agents -------------------------------------------
# Each analyst agent has a specific role and the right tools for their job.

fundamental_analyst = LlmAgent(
    name="FundamentalAnalyst",
    model="gemini-2.5-flash",
    description="Analyzes the financial health and valuation of a company using its stock ticker. Focuses on metrics like P/E ratio, market cap, and dividend yield.",
    instruction="""You are a quantitative financial analyst.
    Your sole responsibility is to analyze a company's financial data.
    Using the `get_financial_data` tool with the provided stock ticker, generate a concise report on the company's financial health.
    Cover key metrics like P/E ratio, dividend yield, and market cap.
    Conclude with a neutral, data-driven summary of the company's valuation and financial stability.""",
    tools=[financial_data_tool],
)

market_sentiment_analyst = LlmAgent(
    name="MarketSentimentAnalyst",
    model="gemini-2.5-flash",
    description="Researches and reports on the market sentiment, news, and public perception of a company.",
    instruction="""You are a market sentiment analyst.
    Your job is to gauge the market's perception of a company by searching for recent news, analyst ratings, and social media discussions.
    Use the `google_search` tool to find relevant information.
    Synthesize your findings into a summary of the current market narrative, identifying key positive and negative sentiment drivers.""",
    tools=[google_search],
)

economic_and_industry_analyst = LlmAgent(
    name="EconomicAndIndustryAnalyst",
    model="gemini-2.5-flash",
    description="Analyzes the broader economic and industry-specific trends affecting a company.",
    instruction="""You are a macroeconomic and industry strategist.
    Your task is to analyze the industry and macroeconomic landscape for a given company.
    Use the `google_search` tool to research the company's industry, its competitors, and relevant economic factors (e.g., interest rates, sector growth trends).
    Provide a report on whether the industry is facing tailwinds or headwinds and how the company is positioned within its competitive landscape.""",
    tools=[google_search],
)


# --- 3. Define the Coordinator Agent ---------------------------------------
# The CIO agent orchestrates the specialists by using them as tools.

chief_investment_officer = LlmAgent(
    name="ChiefInvestmentOfficer",
    model="gemini-2.5-pro",
    description="A top-level agent that analyzes stocks for long-term potential by coordinating a team of specialist analysts.",
    instruction="""You are a Chief Investment Officer managing a team of financial analysts.
    Your goal is to form a comprehensive investment thesis on a company based on a user's request.

    To do this, you must delegate tasks to your specialist analysts by calling them as tools in the following order:
    1.  Call the `FundamentalAnalyst` with the stock ticker to get the company's financial health.
    2.  Call the `MarketSentimentAnalyst` with the company name to understand public perception and news.
    3.  Call the `EconomicAndIndustryAnalyst` with the company name and its industry to get the big-picture context.

    After receiving reports from all three analysts, synthesize their findings into a final, coherent investment thesis.
    Your final report should include:
    - A brief overview of the company.
    - A summary of the fundamental analysis.
    - A summary of the market sentiment.
    - A summary of the industry and economic outlook.
    - A concluding paragraph with your overall investment thesis, including potential upsides and risks.

    Do not perform any analysis yourself. Your role is to delegate, synthesize, and present the final report.""",
    tools=[
        AgentTool(agent=fundamental_analyst),
        AgentTool(agent=market_sentiment_analyst),
        AgentTool(agent=economic_and_industry_analyst),
    ],
)

# Set the root agent for the application
root_agent = chief_investment_officer


# --- 4. Runner and Execution -----------------------------------------------
# This part sets up the ADK runner to execute the agent system.

async def main():
    """
    Initializes and runs the agentic system with a sample query.
    """
    # You may need to set your GOOGLE_API_KEY as an environment variable
    # if it's not already configured in your environment.
    if not os.environ.get("GOOGLE_API_KEY"):
        print("🔴 GOOGLE_API_KEY environment variable not set.")
        print("Please set it to run the agent.")
        return

    print("🟢 Initializing Stock Analysis Agentic System...")

    # Initialize services for the runner
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="stock_analysis_app",
        agent=root_agent,
        session_service=session_service,
    )

    # Create a session for the interaction
    session = await session_service.create_session(user_id="test_user")

    # The user's query
    # Try other tickers like 'MSFT', 'NVDA', 'TSLA'
    user_query = "Should I invest in Google (GOOGL) for the long term?"
    print(f"\n💬 User Query: {user_query}")
    print("-" * 30)

    # Create the initial message content
    message = types.Content(role="user", parts=[types.Part(text=user_query)])

    # Run the agent and stream the events
    print("🧠 ChiefInvestmentOfficer is starting the analysis...\n")
    try:
        events_async = runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=message
        )
        final_response = ""
        async for event in events_async:
            if event.type == "thought":
                print(f"🤔 Thinking: {event.thought.text}")
            if event.type == "tool_code":
                print(f"🛠️ Calling Tool: {event.tool_code.code}")
            if event.type == "tool_output":
                # Don't print the full tool output as it can be very long
                print(f"✅ Tool Finished: {event.tool_output.name}")
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
                        final_response += part.text
        print("\n\n" + "-" * 30)
        print("✅ Analysis Complete.")

    except Exception as e:
        print(f"\n\n🔴 An error occurred during the agent run: {e}")


if __name__ == "__main__":
    # To run this code:
    # 1. Make sure you have the ADK installed.
    # 2. Install yfinance: pip install yfinance
    # 3. Set your GOOGLE_API_KEY environment variable.
    #    For example: export GOOGLE_API_KEY="YOUR_API_KEY"
    asyncio.run(main())
