from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# from langchain_community.llms.ollama import Ollama

llm = ChatOpenAI(model="gpt-3.5-turbo")
# llm = Ollama(model="llama3") # for Ollama users

# Step 1
# Define the system prompt message
system_prompt_message = SystemMessage(
    content="""
System Prompt: As a technical support specialist, provide clear and concise instructions or solutions tailored to the user's problem.
"""
)

#  Define the user's problem message
user_problem_message = HumanMessage(
    content="""
    User's Problem: My computer is running slow and freezing frequently. What should I do? """
)
# Generate responses using the ChatOpenAI object
response_with_role_playing = llm.invoke(
    [system_prompt_message, user_problem_message])

print(response_with_role_playing)


# # Step 2
# # Comment out the previous step.
# # Define prompts with and without SQL schema
# prompt_with_schema = """
# System Prompt: Write a SQL query to retrieve all employees from the 'employees' table who have a salary greater than $50,000.
# SQL Schema:
# Table: employees_table
# Columns: id (INTEGER), name (TEXT), salary
# (INTEGER)
# """
# prompt_without_schema = """
# System Prompt: Write a SQL query to retrieve all employees who have a salary greater than $50,000.
# """
# # Generate responses using OpenAI
# response_with_schema = llm.invoke(prompt_with_schema).content

# response_without_schema = llm.invoke(prompt_without_schema).content

# print(response_with_schema)
# print(response_without_schema)
