import json

from homework.adapters import get_adapter
from homework.summarizer import summarize_conversation
from homework.few_shot_messages import base_messages

system_prompt =  {
        "content": "You are an software architect specializing in CRUD applications.",
        "participant_type": "system"
}

# Initialize the chat models
openai_adapter = get_adapter('openai')
openai_adapter.messages = [
    openai_adapter.base_message_to_llm_message(system_prompt)
]
ollama_adapter = get_adapter('ollama')
ollama_adapter.messages = [
    ollama_adapter.base_message_to_llm_message(system_prompt)
]

ai, human = openai_adapter, ollama_adapter
conversation_history = []

def log_message(message, conversation_log=[]):
    conversation_log.append({
        "content": message["content"],
        "llm_model": message["llm_model"]
    })

for message in base_messages:
    ai.messages.append(ai.base_message_to_llm_message({ "content": message['content'], "participant_type": 'ai'}))
    log_message({"content": message['content'], "llm_model": ai.llm_model}, conversation_history)
    human.messages.append(human.base_message_to_llm_message({"content": message['content'], "participant_type": 'human'}))
    ai, human = human, ai

summary_frequency = 5  # Summarize every 3 exchanges
# max_context_length = 8  # Keep the most recent 6 messages for context
max_messages = 15  # End conversation after 13 messages


for idx in range(max_messages):
    next_base_message = ai.to_base_message(ai.invoke(ai.messages))
    log_message(next_base_message, conversation_history)
    ai.messages.append(ai.base_message_to_llm_message({"content": next_base_message['content'], "participant_type": 'ai'}))
    human.messages.append(human.base_message_to_llm_message({"content": next_base_message['content'], "participant_type": 'human'}))
    
    # Proactive context management: Summarize every few exchanges
    if len(conversation_history) % summary_frequency == 0:
        summary = summarize_conversation(conversation_history, ai.llm_model)
        log_message(summary, conversation_history)
        ai.messages.append(ai.base_message_to_llm_message({"content": summary['content'], "participant_type": 'ai'}))
        human.messages.append(human.base_message_to_llm_message({"content": summary['content'], "participant_type": 'human'}))

        # Maintain a rolling window of the most recent messages
        # if len(conversation_history) > max_context_length:
        #     conversation_history = conversation_history[-max_context_length:]
    
    ai, human = human, ai

    # End the conversation after 15 messages
    if len(conversation_history) >= max_messages:
        print("Conversation ended.")
        break

    # Write the conversation log to a JSON file
with open('./homework/conversation_log.json', 'w') as f:
    json.dump(conversation_history, f, indent=4)
