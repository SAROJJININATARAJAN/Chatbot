import streamlit as st
import httpx
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import HumanMessage, AIMessage
from langchain.agents.agent import AgentExecutor

# 🔑 OpenAI API key and endpoint
API_KEY = "sk-JmVEeaH6p90azCuJwyliJQ"
BASE_URL = "https://genailab.tcs.in/"
MODEL_NAME = "azure/genailab-maas-gpt-4o"

# 🧠 Custom HTTP client
client = httpx.Client(verify=False)

# 🧠 Initialize ChatOpenAI
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    http_client=client,
    temperature=0.7
)

# 🛠️ Define tools the agent can use
def calculator_tool(input: str) -> str:
    try:
        result = eval(input)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for math calculations. Input should be a valid Python expression like '2 + 2' or '3 * (4 + 5)'"
    )
]

# 🤖 Initialize agent
agent: AgentExecutor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# 🌐 Streamlit UI
st.set_page_config(page_title="Agent Chatbot", page_icon="🧠")
st.title("🧠 Chat with Task Agent")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me to calculate or solve something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        reply = agent.run(prompt)
    except Exception as e:
        reply = f"⚠️ Agent failed: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
