from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas,SQL,Power BI,Excel are tools used for data analysis.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings()

vector_store = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    persist_directory= "Lecture-02/chroma_db"    # for Locally Storing Data
)

result = vector_store.similarity_search("What are the tools used for Data Analysis", k =2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vector_store.as_retriever()

doc = retriever.invoke("What is Deep Learning")

for d in docs:
    print(d.page_content)