import os
import asyncio
import yfinance as yf
from typing import Optional, Dict

# Import Pydantic for explicit data modeling
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, AgentTool, google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- 1. Define an Explicit Data Model for the Tool's Output ---
# This resolves the parsing error by giving the ADK a clear schema for the
# tool's return value, instead of a generic Dict.
class FinancialData(BaseModel):
    """Data model for the financial information of a stock."""
    company_name: Optional[str] = Field(default=None, description="The full name of the company.")
    sector: Optional[str] = Field(default=None, description="The sector the company operates in.")
    industry: Optional[str] = Field(default=None, description="The specific industry of the company.")
    market_cap: Optional[float] = Field(default=None, description="The market capitalization of the company.")
    pe_ratio: Optional[float] = Field(default=None, description="The trailing Price-to-Earnings (P/E) ratio.")
    forward_pe: Optional[float] = Field(default=None, description="The forward Price-to-Earnings (P/E) ratio.")
    dividend_yield: Optional[float] = Field(default=None, description="The dividend yield as a percentage.")
    price_to_book: Optional[float] = Field(default=None, description="The Price-to-Book (P/B) ratio.")
    fifty_two_week_high: Optional[float] = Field(default=None, description="The 52-week high stock price.")
    fifty_two_week_low: Optional[float] = Field(default=None, description="The 52-week low stock price.")
    average_volume: Optional[int] = Field(default=None, description="The average daily trading volume.")
    short_summary: Optional[str] = Field(default=None, description="A brief summary of the company's business.")
    error: Optional[str] = Field(default=None, description="An error message if data retrieval fails.")


# --- 2. Define Specialist Tools ---------------------------------------------
# The function now returns the Pydantic model instead of a dict.
def get_financial_data(ticker: str) -> FinancialData:
    """
    Retrieves key financial data for a given stock ticker using yfinance.

    Args:
        ticker: The stock ticker symbol (e.g., 'GOOGL', 'MSFT').

    Returns:
        A FinancialData object containing key financial metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Create a dictionary with the data, matching the Pydantic model fields
        financials_dict = {
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
        # Return an instance of the FinancialData model
        return FinancialData(**financials_dict)
    except Exception as e:
        # In case of an error, return a FinancialData object with the error field populated
        return FinancialData(error=f"Could not retrieve data for {ticker}. Error: {str(e)}")

# Wrap the function in ADK's FunctionTool. It will now correctly infer the schema.
financial_data_tool = FunctionTool(func=get_financial_data)


# --- 3. Define Specialist Agents -------------------------------------------
# The agent definitions remain the same. They are robust to the tool's internal implementation.

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


# --- 4. Define the Coordinator Agent ---------------------------------------
# The CIO agent's definition also remains unchanged.

chief_investment_officer = LlmAgent(
    name="ChiefInvestmentOfficer",
    model="gemini-2.5-pro", # Use a more powerful model for reasoning and synthesis
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


# --- 5. Runner and Execution -----------------------------------------------
# This part remains the same.

async def main():
    """
    Initializes and runs the agentic system with a sample query.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        print("🔴 GOOGLE_API_KEY environment variable not set.")
        print("Please set it to run the agent.")
        return

    print("🟢 Initializing Stock Analysis Agentic System...")

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="stock_analysis_app",
        agent=root_agent,
        session_service=session_service,
    )

    session = await session_service.create_session(user_id="test_user")

    user_query = "Should I invest in Microsoft (MSFT) for the long term?"
    print(f"\n💬 User Query: {user_query}")
    print("-" * 30)

    message = types.Content(role="user", parts=[types.Part(text=user_query)])

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
    # 2. Install yfinance and pydantic: pip install yfinance pydantic
    # 3. Set your GOOGLE_API_KEY environment variable.
    asyncio.run(main())
