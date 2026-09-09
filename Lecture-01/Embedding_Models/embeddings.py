from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

texts = ["Hello Saksham this side",
         "I am learning Gen AI",
         "I am transitioning from Data Science",
         "Will make some solid projects on Generative AI"]
vector = embedding.embed_documents(texts)

print(vector)