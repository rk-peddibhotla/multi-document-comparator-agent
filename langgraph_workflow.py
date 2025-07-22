from langgraph import Workflow, CodeNode


def hello_node():
    return "Hello from LangGraph!"

with Workflow() as wf:
    greet = CodeNode(func=hello_node, name="GreetingNode")

if __name__ == "__main__":
    result = wf.run()
    print(result[greet])