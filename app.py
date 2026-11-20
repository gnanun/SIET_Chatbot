
import streamlit as st
from rag_engine import load_base, ask_rag

st.set_page_config(page_title="SIET_Chatbot", layout="centered")
st.title("Shridevi Institute of Engineering and Technology - ChatBot")
st.caption("Ask anything about college information....")
st.caption("Created by Gnanesh N")

mykey = st.secrets["GROQ_API_KEY"]


# Initialize RAG engine
@st.cache_resource
def initiate_rag():
    return load_base("01_College_Information.pdf")


# Load index, embedding model and chunks
index, embed_model, chunks = initiate_rag()


# Initialize chat history
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []


# Display previous chat messages
for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])


# Accept user question
question = st.chat_input("Ask anything about college....")


if question:

    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Asking RAG for answer...."):

            answer = ask_rag(
                question,
                embed_model,
                chunks,
                index,
                api_key=mykey
            )

            st.write(answer)

            st.session_state.chat_hist.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


# Clear chat history
btn = st.button("Clear Chat")

if btn:
    st.session_state.chat_hist = []
    st.success("Chat History Deleted Successfully")
    st.rerun()
