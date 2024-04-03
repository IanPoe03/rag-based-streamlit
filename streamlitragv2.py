#streamlitragv2 - currently, it allows user to upload a file and then answers questions based on embeddings of this file


import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS



#Main part of the page - it displays the starting screen and chat interface
st.title('RAG-Based Virtual Chatbot')
st.caption('Hi! I can answer questions based on documents provided to me :)')


#Sidebar widget for API key entry
with st.sidebar:
    user_text_box = st.text_input("Enter OpenAI API Key", key = "api_key", type = "password")
    OPENAI_API_KEY = user_text_box
    if OPENAI_API_KEY:
        st.write("Successfully Applied OpenAI API Key")

# #Sidebar widet for uploading documents - currently only accepts .txt
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a .txt file to my knowledge", type='txt')
    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue().decode("utf-8")
        st.write(f"Processing file ...")
        #chunks are length 512, overlap 128 - this performs well so far
        chunker = CharacterTextSplitter(chunk_size = 512, chunk_overlap = 128)
        contents = chunker.split_text(file_contents)   
        embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        vector_space = FAISS.from_texts(contents, embeddings)
        st.write("Chunked and Embedded") 

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
    
    #check that uploaded file has been embedded
    if not (vector_space):
        st.info("Please upload a file to continue.")
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


