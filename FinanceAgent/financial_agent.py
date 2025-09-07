from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
#phi_api_key = os.getenv("PHI_API_KEY")

Groq.api_key = groq_api_key

#This is a web search agent
websearch_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources of your information."],
    show_tool_calls=True,
    markdown=True

)

#websearch_agent.print_response("What is the latest news about OpenAI?", stream=True)


#Financial agent
finance_agent=Agent(
    name="Financial Agent",
    role="Act as a financial advisor to get financial data",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use table to display the data"],
    show_tool_calls=True,
    markdown=True

)

# Test financial agent directly
#finance_agent.print_response("Get analyst recommendations and latest news for NVDA", stream=True)


multi_ai_agent=Agent(
    team=[websearch_agent, finance_agent],
    name="Multi AI Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recomendations and share the latest news for NVDA",stream=True)