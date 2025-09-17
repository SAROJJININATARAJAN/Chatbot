import streamlit as st
from openai import AzureOpenAI

# 🔑 Azure OpenAI credentials (directly from your details)
endpoint = "https://dev-openai-service-01.openai.azure.com/"
key = "4UZgHRJTSOsNZBtqp8gquNQcwWoCWhlQkuj0kLlPK3P2auvrD5sjJQQJ99BEAC77bzfXJ3w3AAABACOGFl9n"
deployment = "groupb-2d8e1a4f-67b3-49c9-84c7-1a9f9e3f9c22"
api_version = "2025-01-01-preview"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version=api_version
)

# Streamlit app setup
st.set_page_config(page_title="Azure Chatbot", page_icon="🤖")
st.title("🤖 Chat with Azure OpenAI")

# Chat history stored in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ API call failed: {e}"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
