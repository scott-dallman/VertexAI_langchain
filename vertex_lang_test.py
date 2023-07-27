import google.auth
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
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo)
from langchain.agents import initialize_agent, AgentType

credentials, project_id = google.auth.default()

llm = VertexAI(temperature=0.1,  top_p=40, top_k=0.95)

def load_document_vectorstore(document):
    embeddings = VertexAIEmbeddings()
    # Create and load PDF
    loader = PyPDFLoader(document)
    # Split pages from PDF
    pages = loader.load_and_split()
    # Load documents into vector database ChromaDB
    store = Chroma.from_documents(documents=pages, embedding=embeddings, collection_name="google_report")
    return store


store = load_document_vectorstore("20230426_alphabet_10Q.pdf")

# Create vectorstore info object - metadata repo
vectorstore_info = VectorStoreInfo(
    name="alphabet_10Q",
    description="a quartly report as a pdf",
    vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

# agent_executor = create_vectorstore_agent(
#     llm=llm,
#     toolkit=toolkit.get_tools(),
#     verbose=True
# )

agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

prompt = "What was google revenue for 2022?"

response = agent.run(prompt)

print(response)