# Vertex_langchain

Python code creates a demo of different ways to use Vertex LLMs.  It uses langchain and chromadb to vectorize PDF documents and provide responses to questions.

For authentication to vertex you must have set your gcloud profile and run the below command to isses credentials 

run "gcloud auth application-default login" to produce the creditionals used to access the Vertex API


--------------

To run the application after authentication:

streamlit run app.py
