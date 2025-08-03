import streamlit as st
import os, runpy, re
import matplotlib.pyplot as plt
import yfinance as yf

# ✅ Helper: Run a notebook/py file
def run_notebook(ipynb_path):
    if ipynb_path.endswith(".ipynb"):
        py_path = ipynb_path.replace(".ipynb", ".py")
        os.system(f'jupyter nbconvert --to script "{ipynb_path}" --output "{os.path.splitext(os.path.basename(py_path))[0]}"')
        namespace = runpy.run_path(py_path)
    else:
        namespace = runpy.run_path(ipynb_path)
    return namespace

# ✅ Load RAG (Livermore psychology)
rag_ns = run_notebook("modules/livermore_rag_setup.py")
get_livermore_answer = rag_ns['get_livermore_answer']

# ✅ Load professor's stock analysis
prof_ns = run_notebook("modules/JesseLivermore_Assignment_Final.py")
get_stock_signal = prof_ns['get_stock_signal']

# ✅ Extract ticker
def extract_ticker(question):
    match = re.search(r"\b[A-Z]{2,5}\b", question)
    return match.group(0) if match else None

# ✅ Generic Stock Analysis
def get_generic_stock_analysis(ticker):
    data = get_stock_signal(ticker)
    return f"""
    📈 **Generic Stock Analysis**
    - Ticker: {ticker}
    - Current Price: {data['price']}
    - Technical Signal: {data['signal']}
    """

# ✅ Livermore Commentary
def get_livermore_commentary(ticker):
    signal_data = get_stock_signal(ticker)
    prompt = f"""
    You are Jesse Livermore. Based on this stock data, give your general take:

    Ticker: {ticker}
    Current Price: {signal_data['price']}
    Technical Signal: {signal_data['signal']}

    Provide a **generic trading commentary** with psychology insights.
    """
    return get_livermore_answer(prompt)

# ✅ Chatbot Logic
def chat_with_livermore(question):
    ticker = extract_ticker(question)
    if ticker:
        signal_data = get_stock_signal(ticker)
        stock_context = f"""
        [Stock Analysis]
        Ticker: {ticker}
        Current Price: {signal_data['price']}
        Technical Signal: {signal_data['signal']}
        """
        prompt = f"""
        You are Jesse Livermore.

        Use the stock data below and your trading principles to answer.

        {stock_context}

        Question: {question}

        🔹 Give a strict recommendation (BUY/SELL/HOLD).
        """
        return get_livermore_answer(prompt)
    return get_livermore_answer(question)

# ✅ Streamlit UI Config
st.set_page_config(layout="wide", page_title="Jesse Livermore Trading App")

# Sidebar - Dropdown
st.sidebar.header("📊 Select Stock")
tickers = ["AMZN", "AAPL", "MSFT", "TSLA", "META", "GOOGL", "NVDA"]
selected_ticker = st.sidebar.selectbox("Choose a stock:", tickers)

# Livermore Commentary below dropdown
st.sidebar.subheader(f"🧠 Livermore's Commentary on {selected_ticker}")
st.sidebar.markdown(get_livermore_commentary(selected_ticker))

# Layout: Chart + Chatbot
col1, col2 = st.columns([1.2, 1.8])

# --- Chart Section (Non-Scrollable) ---
with col1:
    st.subheader(f"📈 {selected_ticker} - Livermore Strategy Chart")
    df = yf.download(selected_ticker, period="18mo", interval="1d")
    df['50MA'] = df['Close'].rolling(window=50).mean()
    df['200MA'] = df['Close'].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df['Close'], label="Close Price")
    ax.plot(df['50MA'], label="50MA")
    ax.plot(df['200MA'], label="200MA")
    ax.set_title(f"{selected_ticker} Price & Moving Averages")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📌 Stock Insights")
    st.markdown(get_generic_stock_analysis(selected_ticker))

# --- Chatbot Section (Scrollable) ---
with col2:
    st.markdown("<h2 style='text-align:center;'>💬 Livermore Chatbot</h2>", unsafe_allow_html=True)

    # Chat history init
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    # Scrollable chat window
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"🧑‍💻 **You:** {msg['content']}")
            else:
                st.markdown(f"🎩 **Livermore:** {msg['content']}")

    # Input at bottom with limited width
    user_input = st.text_input("Type your question and press Enter:", key="chat_input")

    # ✅ Only process if new question
    if user_input and user_input != st.session_state.last_question:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = chat_with_livermore(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_question = user_input
        st.rerun()
