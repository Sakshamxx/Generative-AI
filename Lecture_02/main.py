from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

emb_model = MistralAIEmbeddings()

vector_store = Chroma(
    persist_directory="Lecture_02/Chroma_db",
    embedding_function= emb_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

LLM = ChatMistralAI(model = "ministral-3b-2512")

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG SYSTEM CREATED")

print("Press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })
    
    response = LLM.invoke(final_prompt)
    print(f"\n AI: {response.content}")