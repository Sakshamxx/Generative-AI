from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 10
    },
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is data science?")

print(response.content)