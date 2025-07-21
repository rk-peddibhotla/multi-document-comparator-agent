from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama
from langchain.tools import Tool
from tools import get_system_info  # basic function

# Wrap the function to accept a single input string
wrapped_system_info = lambda _: get_system_info()

# Define the tool properly
tools = [
    Tool(
        name="get_system_info",
        func=wrapped_system_info,
        description="Returns system info like OS, CPU, RAM.",
        return_direct=True  # important for final answer to not conflict with tool call
    )
]

# Load the LLM
llm = ChatOllama(model="gemma:2b")

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=False,  # Clean output
    handle_parsing_errors=True
)

def main():
    print("Welcome to Smart CLI Assistant powered by Gemma 2B + Tools!")
    while True:
        query = input("\nAsk me anything (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = agent.run(query)
            print("\nGemma says:\n", response)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
