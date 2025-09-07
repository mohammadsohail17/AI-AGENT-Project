from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground,serve_playground_app

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
phi.api=os.getenv("PHI_API_KEY")



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
    description="You are an finance analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True

)

app=Playground(agents=[websearch_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)

