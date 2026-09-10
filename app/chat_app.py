import streamlit as st
import sys
sys.path.append("src")
from agent import agent

st.title("🌦️ Monsoon Intelligence Assistant")
st.caption("Ask about monsoon science, live weather, or get rainfall predictions")

if "messages" not in st.session_state:
	st.session_state.messages = []
if "thread_id" not in st.session_state:
	st.session_state.thread_id = "streamlit_session"

for msg in st.session_state.messages:
	with st.chat_message(msg["role"]):
		st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)

	config = {"configurable": {"thread_id": st.session_state.thread_id}}
	with st.chat_message("assistant"):
		with st.spinner("Thinking..."):
			result = agent.invoke({"messages": [("user", prompt)]}, config)
			response = result["messages"][-1].content
			st.markdown(response)

	st.session_state.messages.append({"role": "assistant", "content": response})