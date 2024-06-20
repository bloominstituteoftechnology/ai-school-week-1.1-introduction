from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

messages = [
    HumanMessage(
        content="Translate this sentence from English to French: I love programming."
    ),
    AIMessage(content="J'adore la programmation."),
    HumanMessage(content="What did you just say?"),
]

result = chat.invoke(messages).content

messages.append(AIMessage(content=result))
print(messages)

# Continue the conversation
# Try: Can you translate that phrase into Spanish?

while True:
    prompt = input("Prompt: ")
    messages.append(HumanMessage(content=prompt))

    result = chat.invoke(messages).content

    messages.append(AIMessage(content=result))
    print(messages)
