from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama

llm_openai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
llm_ollama = Ollama(model="llama3")  # for Ollama users

system_prompt = "You are a software engineer discussing the different approaches to building a CRUD application. You are in a conversation with another software engineer that may or may not have different perspectives on this. If you agree on something, move on to a different topic or ask a question to keep the conversation going and interesting."

openai_messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="What is the best way to build a CRUD application?")
]

ollama_messages = [{
    "role": "system",
    "content": system_prompt,
}
]

# What is the initial message?
initial_message = llm_openai.invoke(openai_messages).content
print("initial_message: ", initial_message)
openai_messages.append(AIMessage(content=initial_message))
ollama_messages.append({
    "role": "user",
    "content": initial_message,
})

total_messages = 1

while total_messages < 20:
    ollama_result = llm_ollama.invoke(ollama_messages)
    print("ollama_result: ", ollama_result)
    ollama_messages.append({
        "role": "assistant",
        "content": ollama_result
    })

    openai_messages.append(HumanMessage(content=ollama_result))
    openai_result = llm_openai.invoke(openai_messages).content
    print("openai_result: ", openai_result)
    openai_messages.append(AIMessage(content=openai_result))

    ollama_messages.append({
        "role": "user",
        "content": openai_result,
    })

    total_messages += 1
