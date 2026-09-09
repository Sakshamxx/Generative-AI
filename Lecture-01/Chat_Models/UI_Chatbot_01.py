import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sassy Baddie AI",
    page_icon="💅",
    layout="centered",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = "You are an Sassy Baddie AI Agent"

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #2b0f3a 0%, #1a0b2e 40%, #0d0518 100%);
        color: #f5f0ff;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Title */
    .hero-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff5fa2, #b967ff, #5fd6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        text-align: center;
        color: #c9b8e8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles */
    .chat-row {
        display: flex;
        margin: 10px 0;
        align-items: flex-end;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.bot { justify-content: flex-start; }

    .bubble {
        max-width: 75%;
        padding: 12px 16px;
        border-radius: 18px;
        line-height: 1.45;
        font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .bubble.user {
        background: linear-gradient(135deg, #ff5fa2, #b967ff);
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble.bot {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        color: #f5f0ff;
        border-bottom-left-radius: 4px;
    }

    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        margin: 0 8px;
        flex-shrink: 0;
    }
    .avatar.user { background: linear-gradient(135deg, #5fd6ff, #b967ff); }
    .avatar.bot { background: linear-gradient(135deg, #ff5fa2, #ffb85f); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0b2e, #0d0518);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * { color: #f5f0ff !important; }

    /* Chat input box */
    .stChatInput textarea, div[data-testid="stChatInput"] textarea {
        background-color: rgba(255,255,255,0.06) !important;
        color: #f5f0ff !important;
        border-radius: 14px !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff5fa2, #b967ff);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💅 Sassy Baddie AI")
    st.caption("Powered by Mistral · ministral-3b-2512")
    st.divider()

    temperature = st.slider("Sass Level 🔥 (temperature)", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.slider("Response Length (max tokens)", 50, 500, 100, 10)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        st.rerun()

    st.divider()
    st.caption("Type **0** in the chat to end the conversation, just like the original CLI script.")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

if "ended" not in st.session_state:
    st.session_state.ended = False

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">💅 Sassy Baddie AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Welcome to the Chatbot, how may I help you, Sir 😏</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "bot"
    avatar = "🧑" if role == "user" else "💋"
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="bubble user">{msg.content}</div>
                <div class="avatar user">{avatar}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row bot">
                <div class="avatar bot">{avatar}</div>
                <div class="bubble bot">{msg.content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------------
if st.session_state.ended:
    st.info("Chat ended. Click **Clear Chat** in the sidebar to start a new conversation. 💋")
else:
    prompt = st.chat_input("Spill the tea, Sir...")

    if prompt is not None:
        st.session_state.messages.append(HumanMessage(content=prompt))

        if prompt.strip() == "0":
            st.session_state.ended = True
            st.rerun()
        else:
            model = ChatMistralAI(
                model="ministral-3b-2512",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            with st.spinner("Thinking of something sassy to say... 💅"):
                response = model.invoke(st.session_state.messages)

            st.session_state.messages.append(AIMessage(content=response.content))
            st.rerun()