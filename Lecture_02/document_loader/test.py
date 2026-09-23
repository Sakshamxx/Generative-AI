from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap = 2
)

data = TextLoader("Lecture-02/document_loader/notes.txt")

docs = data.load()
chunks = splitter.split_documents(docs)
# print(docs[0].page_content)
# print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()