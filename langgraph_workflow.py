from langgraph import Workflow, CodeNode

# A simple node that returns a greeting
def hello_node():
    return "Hello from LangGraph!"

with Workflow() as wf:
    greet = CodeNode(func=hello_node, name="GreetingNode")

if __name__ == "__main__":
    result = wf.run()
    print(result[greet])
