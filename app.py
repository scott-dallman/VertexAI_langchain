"""Python code the creates a demo of different ways to use Vertex LLMs
uses langchain and chromadb to vectorize PDF documents

Author: Scott Dallman
"""

# imports
from PIL import Image
import streamlit as st
import pandas as pd

# Import langchain vertex AI implementation
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Import PDF document loaders
from langchain_community.document_loaders import PyPDFLoader

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Import vector stores
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
from langchain.agents import (
    initialize_agent,
    AgentType,
)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.set_page_config(layout="wide")


@st.cache_resource
def load_lang_model():
    """Loading the vertex model"""
    return VertexAI(model_name="gemini-pro")


llm = load_lang_model()


with st.sidebar:
    image = Image.open("vertex-ai.png")
    st.image(image)
    st.title("Demo Vertex AI LLM")
    st.info(
        "This this a small demo using Streamlit, \
            Vertex AI and displays whats possible with LLM's"
    )
    choice = st.radio(
        "Navigation to LLM models", ["Create", "Summarize", "Discover", "Automate"]
    )


# The selection to create content from a LLM
if choice == "Create":
    st.title("Create with Gen AI")
    st.markdown(
        ":blue[Transform your creative process with \
                the power of GenAI! Eliminate writer’s \
                block & boost productivity by generating writing]"
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_create = st.chat_input(
        "Ask a question to generate ideas on a topic:", key="prompt_create"
    )

    # Create the LLM chain and memory buffer
    TEMPLATE = """You are a chatbot having a conversation with a human.
    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=TEMPLATE
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    # # Testing out conversation Chain
    # memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    # conversation = ConversationChain(
    #     llm=llm,
    #     verbose=True,
    #     memory=memory
    # )

    # Response from LLM
    if prompt_create:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt_create)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_create})

        response = llm_chain.predict(human_input=prompt_create)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# The selection to summarize content from a LLM
if choice == "Summarize":
    st.title("Summarize with Gen AI")
    st.markdown(
        ":blue[Take long form chats, emails, or reports and \
                distill them down to their core for quick comprehension.]"
    )
    long_text = st.text_area(
        "Enter your text to be summarized:", key="long_text", height=300
    )

    TEMPLATE = """Please summarize the text for me \
        in a few sentences : {long_text} """

    prompt_summary = PromptTemplate(template=TEMPLATE, input_variables=["long_text"])
    llm_chain = LLMChain(prompt=prompt_summary, llm=llm)

    if len(long_text) > 100:
        if st.button("Generate Summary"):
            response = llm_chain.run(long_text)
            st.info(response)
            long_text = st.empty
    else:
        st.warning(
            "The text is not long enough must\
                   be at least 100 characters"
        )


# Function to load the pdf document
@st.cache_resource()
def load_document_vectorstore(document):
    """loading the vectorstore documents"""
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    # Create and load PDF
    loader = PyPDFLoader(document)
    # Split pages from PDF
    pages = loader.load_and_split()
    # Load documents into vector database ChromaDB
    chroma_store = Chroma.from_documents(
        documents=pages, embedding=embeddings, collection_name="google_report"
    )
    return chroma_store


# The selection to discover content from document using a LLM
if choice == "Discover":
    st.title("Discover with Gen AI")
    st.markdown(
        ":blue[Build AI enhanced search engines \
                or assistive experiences to help customers \
                navigate complex transactions or analyze \
                patterns in documents.]"
    )

    store = load_document_vectorstore("20230426_alphabet_10Q.pdf")

    # Create vectorstore info object - metadata repo
    vectorstore_info = VectorStoreInfo(
        name="alphabet_10Q", description="a quartly report as a pdf", vectorstore=store
    )

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

    vector_agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_discover = st.text_input(
        "Ask a question about the doc", key="prompt_discover"
    )

    if prompt_discover:
        with st.expander("Document Similarity Search"):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt_discover)
            st.write(search[0][0].page_content)
        response = vector_agent.run(prompt_discover)
        st.info(response)


# The selection to automate content from a LLM
if choice == "Automate":
    st.title("Automate with Gen AI")
    st.markdown(
        ":blue[Transform from time consuming, \
                expensive analytics processes to efficient ones.]"
    )
    st.markdown("Dataframe with small Titanic dataset loaded ")

    df = pd.read_csv("Titanic-Dataset.csv")
    st.dataframe(df)
    df_agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True)

    prompt_choice = st.text_input("Ask a question about the doc", key="prompt_choice")

    if prompt_choice:
        response = df_agent.run(prompt_choice)
        st.info(response)
