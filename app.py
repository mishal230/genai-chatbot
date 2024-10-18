import streamlit as st
from transformers import pipeline

# Initialize the text generation pipeline with the DialoGPT model
pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Title of the app
st.title("Chatbot using DialoGPT")

# Initialize session state to hold conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = ""

# Function to generate a response
def generate_response(user_input):
    # Update conversation history
    st.session_state.conversation_history += f"User: {user_input}\n"
    
    # Generate a response
    response = pipe(st.session_state.conversation_history, max_length=150, num_return_sequences=1)
    bot_response = response[0]['generated_text'].split("User:")[-1].strip()
    
    # Update conversation history with bot response
    st.session_state.conversation_history += f"Bot: {bot_response}\n"
    
    return bot_response

# Text input for user to enter their message
user_input = st.text_input("You:", "")

# Button to submit the input
if st.button("Send"):
    if user_input:
        bot_reply = generate_response(user_input)
        st.text_area("Chat History", value=st.session_state.conversation_history, height=300)
    else:
        st.warning("Please enter a message!")

# Display chat history
st.text_area("Chat History", value=st.session_state.conversation_history, height=300)
