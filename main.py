from langchain import OpenAI, VectorDBQA
from langchain.prompts import PromptTemplate
from langchain.chains import ChatVectorDBChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import pinecone 
import streamlit as st
from streamlit_chat import message


pinecone.init(
    api_key="4550550e-1717-4068-8bd9-2481fa994c5a",  # find at app.pinecone.io
    environment="us-central1-gcp"  # next to api key in console
)

embeddings = OpenAIEmbeddings()
index_name = "cab9bot"

docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)



def loadSupportDBQAChain():
    prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, respond kindly by saying that your knowledge is limited and request the user to contact the Cab9 Support Team, don't try to make up an answer.
    
    {context}

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    params = { "prompt" : PROMPT }
    return VectorDBQA.from_chain_type(llm=OpenAI(model_name='gpt-3.5-turbo'), chain_type="stuff",  vectorstore=docsearch, chain_type_kwargs=params)


def loadKnowledgeAgent():

    qa = loadSupportDBQAChain()

    tools = [
        Tool(
            name = "Cab9 knowledge agent",
            func=qa.run,
            description="answer questions and responds to greetings by introducing itself as an AI for Cab9"
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return initialize_agent(tools, llm=ChatOpenAI(model_name='gpt-3.5-turbo'), agent="conversational-react-description", verbose=True, memory=memory)



agent = loadKnowledgeAgent()


### Streamlit UI
st.set_page_config(page_title="Cab9 GPT", page_icon=":robot:")
st.header("Cab9 GPT")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []



def get_text():
    input_text = st.text_input("You: ", "Hi, what can you do?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = agent.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style='thumbs')