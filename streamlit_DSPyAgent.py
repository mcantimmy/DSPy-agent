import streamlit as st
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch  # Importing BootstrapFewShotWithRandomSearch from the Stanford NLP's library

# Initialize BootstrapFewShotWithRandomSearch with the desired metric and parameters
teleprompter = BootstrapFewShotWithRandomSearch(metric='gsm8k_metric', max_bootstrapped_demos=8, max_labeled_demos=8)

# Function to process the prompt using DSPy
def process_prompt(prompt, variables):
    # Incorporate variables into the prompt if any
    for var_name, var_value in variables.items():
        prompt = prompt.replace(f'${var_name}$', var_value)
    # Compile the prompt using DSPy

    # 1) Declare with a signature.
    classify = dspy.Predict('prompt -> optimized prompt')

    # 2) Call with input argument(s). 
    response = classify(sentence=prompt)

    # Define your training set
    training_set = [
        {"input": "What's the weather like today?", "output": "I'm sorry, as an AI, I don't have real-time capabilities. However, you can check the weather on a weather website or app."},
        {"input": "Set a reminder for my meeting tomorrow at 10 AM.", "output": "I'm sorry, as an AI, I don't have the capability to set reminders."},
        {"input": "Tell me a joke.", "output": "Sure, here's a joke for you: Why don't scientists trust atoms? Because they make up everything!"},
        {"input": "What's the capital of France?", "output": "The capital of France is Paris."},
        {"input": "Translate 'Hello' to Spanish.", "output": "'Hello' translates to 'Hola' in Spanish."},
        # Add more input-output pairs as needed
    ]
    compiled_prompt = teleprompter.compile(response, trainset=training_set)
    return compiled_prompt

# Streamlit app layout
st.title('DSPy Prompt Optimization App')

# User input for the prompt
user_prompt = st.text_area('Enter your prompt:', 'Please design an AI agent that is an expert at designing other AI agents to perform specific tasks.')

# Variable assignments
st.subheader('Variable Assignments')
variables = {}
var_names = st.text_input('Enter variable names (comma-separated):', '')
var_values = st.text_input('Enter corresponding variable values (comma-separated):', '')
if var_names and var_values:
    var_names = var_names.split(',')
    var_values = var_values.split(',')
    variables = dict(zip(var_names, var_values))

# Button to process the prompt
if st.button('Optimize Prompt'):
    optimized_prompt = process_prompt(user_prompt, variables)
    st.text_area('Optimized Prompt:', optimized_prompt, height=300)