import streamlit as st
from matcher import matching

with st.container():
    st.header("Semantic Text Matching")
    question = st.chat_input("Enter question to matching")
    if question:
        with st.chat_message("human"):
            st.write(question)

        questions = matching(question)

        with st.status("Matching ...", expanded=True) as status:
            with st.chat_message("ai"):
                for i, q_pred in enumerate(questions):
                    st.write(f"{i + 1}. {q_pred}")

            status.update(label="Matched!", state="complete", expanded=True)
