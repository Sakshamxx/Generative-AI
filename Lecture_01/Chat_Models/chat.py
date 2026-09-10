from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(
    model="ministral-3b-2512", temperature=0, max_tokens=50
)

response = model.invoke(
    "Write a Poem on AI"
)

print(response.content)