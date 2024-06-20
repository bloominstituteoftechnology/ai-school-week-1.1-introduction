from homework.adapters import get_adapter
 
openai_adapter = get_adapter('openai')
ollama_adapter = get_adapter('ollama')

def summarize_conversation(messages, llm_model):
    sum_of_messages = ""
    for msg in messages:
        sum_of_messages += f"\n{msg['content']} "
    prompt = f"Produce a summary of the following conversation & preface your response with 'Summary so far:': \n{sum_of_messages}"
    if llm_model == "openai":
        llm_message = openai_adapter.base_message_to_llm_message({"content": prompt, "participant_type": "human"})
        summary_response = openai_adapter.invoke([llm_message])
        return openai_adapter.to_base_message(summary_response)
    else:
        llm_message = ollama_adapter.base_message_to_llm_message({"content": prompt, "participant_type": "human"})
        summary_response = ollama_adapter.invoke([llm_message])
        return ollama_adapter.to_base_message(summary_response)