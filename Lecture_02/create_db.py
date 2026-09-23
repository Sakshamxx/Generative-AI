from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
load_dotenv()

data = PyPDFLoader("/Users/sakshamchauhan/Downloads/Github/Gen-AI_Agentic-AI/Lecture_02/document_loader/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()

vector_store = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "Store/Chroma_db"
    
)