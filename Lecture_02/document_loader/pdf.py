from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("Lecture-02/document_loader/GRU.pdf")

docs = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)
chunks = splitter.split_documents(docs)

# print(len(docs))
print(chunks[0].page_content)