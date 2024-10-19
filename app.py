import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to generate response from the chatbot
def generate_response(user_input, history=None):
    if history is None:
        history = torch.zeros((1, 0), dtype=torch.long)  # Initialize empty chat history
    # Encode user input and append it to the history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1)
    
    # Create an attention mask (to handle padding if required)
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    
    # Generate the bot's response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id, 
        attention_mask=attention_mask,
        do_sample=True,           # Enable sampling for more diverse responses
        top_k=50,                 # Keep only the top 50 tokens
        top_p=0.95,               # Nucleus sampling
        temperature=0.7           # Control randomness
    )
    
    # Decode and return the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Initialize Streamlit app layout
st.title("🤖 Customer Service Chatbot")
st.subheader("Professional and User-Friendly AI-Powered Bot")
st.markdown("This chatbot provides quick, helpful responses for customer service inquiries.")

# Add user input box and display past conversation
user_input = st.text_input("You: ", key="input")
if 'history' not in st.session_state:
    st.session_state['history'] = None

# Add a button to reset the conversation
if st.button("🔄 Reset Conversation"):
    st.session_state['history'] = None
    st.success("Conversation has been reset!")

# If the user provides input, generate the bot's response
if user_input:
    try:
        # Generate response and update chat history
        response, st.session_state['history'] = generate_response(user_input, st.session_state['history'])
        
        # Display the conversation in a neat format
        st.write(f"**You:** {user_input}")
        st.write(f"**Bot:** {response}")
        
    except Exception as e:
        # Handle any errors gracefully
        st.error(f"Something went wrong: {str(e)}")

# Optional footer
st.markdown("---")
st.markdown("Powered by [Huggingface](https://huggingface.co/microsoft/DialoGPT-medium) and Streamlit. Built with ❤️ by Mishal Zubair.")
