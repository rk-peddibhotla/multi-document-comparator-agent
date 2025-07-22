from langchain.agents import initialize_agent
from langchain_ollama import ChatOllama
from langchain.tools import Tool
from tools import get_system_info  


wrapped_system_info = lambda _: get_system_info()


tools = [
    Tool(
        name="get_system_info",
        func=wrapped_system_info,
        description="Returns system info like OS, CPU, RAM.",
        return_direct=True  
    )
]


llm = ChatOllama(model="gemma:2b")


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=False,  
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