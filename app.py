import streamlit as st
from rag_chatbot import rag_query

st.title("Medical FAQ Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = rag_query(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
if __name__ == "__main__":
    while True:
        query = input("Ask a question (or 'exit'): ")
        if query.lower() == 'exit':
            break
        print("Answer:", rag_query(query))    