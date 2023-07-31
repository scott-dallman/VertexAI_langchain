import pytest

from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
import pandas as pd
import app

def test_load_lang_model():
    """Test that the load_lang_model function returns a VertexAI model."""
    lm = app.load_lang_model()
    assert isinstance(lm, VertexAI)


def test_create_content():
    """Test that the create_content function generates text."""
    question = "What is the meaning of life?"
    prompt = PromptTemplate(template="Question: {question}", input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=app.load_lang_model())
    response = llm_chain.run(question)
    assert response != ""


def test_summarize_content():
    """Test that the summarize_content function summarizes text."""
    long_text = "This is a long text that I want to summarize."
    prompt = PromptTemplate(template="Please summarize the text for me in a few sentences : {long_text}", input_variables=["long_text"])
    llm_chain = LLMChain(prompt=prompt, llm=app.load_lang_model())
    response = llm_chain.run(long_text)
    assert response != ""


def test_discover_content():
    """Test that the discover_content function discovers content from a document."""
    store = app.load_document_vectorstore("20230426_alphabet_10Q.pdf")
    prompt = "What is the revenue for Google?"
    response = store.similarity_search_with_score(prompt)
    assert response != []


def test_automate_content():
    """Test that the automate_content function automates tasks with a dataframe."""
    df = pd.read_csv("titanic.csv")
    df_agent = app.create_pandas_dataframe_agent(llm=app.load_lang_model(), df=df)
    prompt = "What is the survival rate for women?"
    response = df_agent.run(prompt)
    assert response != ""