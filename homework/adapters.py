from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import ollama

class OpenAIAdapter:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        self.messages = []
        self.llm_model = "openai"

    def base_message_to_llm_message(self, base_message):
        content = base_message["content"]
        participant = base_message["participant_type"]
        if participant == 'human':
            return HumanMessage(content=content)
        elif participant == 'ai':
            return AIMessage(content=content)
        elif participant == 'system':
            return SystemMessage(content=content)
        else:
            raise ValueError("Invalid participant type")

    def invoke(self, messages):
        response = self.chat.invoke(messages)
        return response

    def to_base_message(self, message):
        return {
            "content": message.content,
            "llm_model": self.llm_model
        }

class OllamaAdapter:
    def __init__(self):
        self.chat = ollama.chat
        self.messages = []
        self.llm_model = "ollama"

    def base_message_to_llm_message(self, base_message):
        content = base_message["content"]
        participant = base_message["participant_type"]
        if participant == 'human':
            return {'role': 'user', 'content': content}
        elif participant == 'ai':
            return {'role': 'assistant', 'content': content}
        elif participant == 'system':
            return {'role': 'system', 'content': content}
        else:
            raise ValueError("Invalid participant type")

    def invoke(self, messages):
        response = self.chat(model='llama3', messages=messages)
        return response

    def to_base_message(self, message):
        message_obj = message['message']
        return {
            "content": message_obj['content'],
            "llm_model": self.llm_model
        }

def get_adapter(llm: str):
    if llm == 'openai':
        return OpenAIAdapter()
    elif llm == 'ollama':
        return OllamaAdapter()
    else:
        raise ValueError("Invalid LLM type")
