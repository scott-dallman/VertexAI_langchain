"""Python code the creates a demo of different ways to use Vertex LLMs
uses langchain and chromadb to vectorize PDF documents

Author: Scott Dallman
"""

# imports
from PIL import Image
import streamlit as st
import pandas as pd

# Import langchain vertex AI implementation
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import VertexAIEmbeddings
# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma
# Import vector stores
from langchain.agents.agent_toolkits import (
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.agents import (
    initialize_agent,
    AgentType,
    create_pandas_dataframe_agent
)


@st.cache_resource
def load_lang_model():
    """Loading the vertex model"""
    return VertexAI(temperature=0.1)


llm = load_lang_model()


with st.sidebar:
    image = Image.open('vertex-ai.png')
    st.image(image)
    st.title("Demo Vertex AI LLM")
    st.info("This this a small demo using Streamlit, \
            Vertex AI and displays whats possiable with LLM's")
    choice = st.radio("Navigation to LLM models",
                      ["Create", "Summarize", "Discover", "Automate"])


# The selection to create content from a LLM
if choice == "Create":
    st.title("Create with Gen AI")
    st.markdown(":blue[Transform your creative process with \
                the power of GenAI! Eliminate writer’s \
                block & boost productivity by generating writing]")
    prompt_create = st.text_input(
        'Ask a question to generate ideas on a topic'
    )

    TEMPLATE = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=TEMPLATE, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    if prompt_create:
        response = llm_chain.run(prompt_create)
        st.markdown(response)


# The selection to summarize content from a LLM
if choice == "Summarize":
    st.title("Summarize with Gen AI")
    st.markdown(":blue[Take long form chats, emails, or reports and \
                distill them down to their core for quick comprehension.]")
    long_text = st.text_area('Enter your text to be summarized', height=300)

    TEMPLATE = """Please summarize the text for me \
        in a few sentences : {long_text} """

    prompt_summary = PromptTemplate(
        template=TEMPLATE,
        input_variables=["long_text"]
    )
    llm_chain = LLMChain(prompt=prompt_summary, llm=llm)

    if len(long_text) > 100:
        if st.button("Generate Summary"):
            response = llm_chain.run(long_text)
            st.info(response)
    else:
        st.warning("The text is not long enough must\
                   be at least 100 characters")


# Function to load the pdf document
@st.cache_resource()
def load_document_vectorstore(document):
    """loading the vectorstore documents"""
    embeddings = VertexAIEmbeddings()
    # Create and load PDF
    loader = PyPDFLoader(document)
    # Split pages from PDF
    pages = loader.load_and_split()
    # Load documents into vector database ChromaDB
    chroma_store = Chroma.from_documents(
        documents=pages,
        embedding=embeddings,
        collection_name="google_report")
    return chroma_store


# The selection to discover content from document using a LLM
if choice == "Discover":
    st.title("Discover with Gen AI")
    st.markdown(":blue[Build AI enhanced search engines \
                or assistive experiences to help customers \
                navigate complex transactions or analyze \
                patterns in documents.]")

    store = load_document_vectorstore("20230426_alphabet_10Q.pdf")

    # Create vectorstore info object - metadata repo
    vectorstore_info = VectorStoreInfo(
        name="alphabet_10Q",
        description="a quartly report as a pdf",
        vectorstore=store
    )

    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

    vector_agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
        )

    prompt_discover = st.text_input('Ask a question about the doc')
    if prompt_discover:
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt)
            st.write(search[0][0].page_content)
        response = vector_agent.run(prompt)
        st.info(response)


# The selection to automate content from a LLM
if choice == "Automate":
    st.title("Automate with Gen AI")
    st.markdown(":blue[Transform from time consuming, \
                expensive analytics processes to efficient ones.]")
    st.markdown("Dataframe with small Titanic dataset loaded ")
    df = pd.read_csv("titanic.csv")

    df_agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True)

    prompt_choice = st.text_input('Ask a question about the doc')
    if prompt_choice:
        response = df_agent.run(prompt)
        st.info(response)
