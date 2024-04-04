#streamlit rag v3 - accesses a persistent database of transcripts

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

#Main part of the page - it displays the starting screen and chat interface
st.title('RAG-Based Chatbot for AI in Marketing')
st.caption('Hi! I can answer questions about the AI in Marketing course. To get started, enter your API key in the sidebar')

#Sidebar widget for API key entry
with st.sidebar:
    user_text_box = st.text_input("Enter OpenAI API Key", key = "api_key", type = "password")
    OPENAI_API_KEY = user_text_box
    if OPENAI_API_KEY:
        st.write("Successfully Applied OpenAI API Key")

#If user enters an API key, load persistent directory and set up vector space from it
if OPENAI_API_KEY:
    persist_dir = 'https://raw.githubusercontent.com/IanPoe03/rag-based-streamlit/tree/main/persistent_directory'
    embeddings =  OpenAIEmbeddings()
    vector_space = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

#set up chat interface - conversation between an assistant and a user
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I can answer questions based on documents provided to me :)"}]

#keep a record of conversation by writing it to screen
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#main part of the program - it runs in the middle section
if prompt := st.chat_input():

    #check for API Key
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    #template for assistant = can be customized based on class needs
    #currently set up to serve as a virtual TA for an AI in Marketing Class
    template = """
    You are a helpful virtual assistant for a class focusing on the application of AI in marketing.
    Your task is to answer student questions about the contents of the course using relevant training information.
    If you don't know the answer, don't try to make one up yourself.
    If the answer exists in your pretrained database, but not in vector_space, the rag databse you've been provided, don't report it.
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """

    #create prompt_assistant based on template, history, context, and question
    prompt_assistant = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
        )

    #currently using gpt-3.5-turbo as the llm
    llm_model = ChatOpenAI(temperature=0.7, model_name = "gpt-3.5-turbo", verbose =False)

    #use embeddings from uploaded document as a retriever with 3-Neighbor KNN
    retriever = vector_space.as_retriever(search_type = "similarity", search_kwargs={"k": 3})

    #set up conversation chain with memory of current conversation
    qa = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type='stuff',
    retriever=retriever,
    verbose=False,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt_assistant,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
    )

    #feed user query into conversation chain and get response
    result = qa({"query": prompt})
    answer = result["result"]

    #write user message and AI answer to screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)


