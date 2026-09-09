from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
load_dotenv()

model = ChatMistralAI(
    model="ministral-3b-2512", temperature=0, max_tokens=100
)

print("Choose your AI Mode")
print("Press 1: Sassy")
print("Press 1: Elegant")
print("Press 1: Sarcastic")

choice = int(input("Mode->"))

if choice == 1:
    mode = "You are an Sassy Queen and a 10/10 Baddie, so respond according to it"
elif choice == 2:
    mode = "You are an Elegant and Intelligent Woman, so respond according to it"
elif choice == 3:
    mode = "You are an Funny, Cool and Sarcastic, so respond according to it"

messages = [
    SystemMessage(content=mode)
]
print(" Welcome to the Chatbot, How may i help you Sir")

while(True):
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot:",response.content)
print(messages)