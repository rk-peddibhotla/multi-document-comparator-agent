from langchain.llms import Ollama

def test_ollama():
    llm = Ollama(model="gemma:2b")
    prompt = "List three key differences between cats and dogs."
    response = llm(prompt)
    print("LLM Response:\n", response)

if __name__ == "__main__":
    test_ollama()
