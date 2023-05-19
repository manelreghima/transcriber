import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

# Define Streamlit app
st.title("Youtube Summarizer Demo")
st.write("This demo uses a chat model to answer questions about a YouTube video.")

# Display video selection and question inputs
YT_video = st.text_input("YouTube Video ID (e.g., 1egAKCKPKCk?t=2743)")
query = st.text_input("Enter your question")

# Process user inputs
if st.button("Get Answer"):
    if not YT_video:
        st.warning("Please enter a YouTube video ID.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        # Load documents (or transcribe) with YoutubeLoader
        loader = YoutubeLoader(video_id=YT_video, language="en")
        yt_docs = loader.load_and_split()
        embeddings = OpenAIEmbeddings()
        yt_docsearch = Chroma.from_documents(yt_docs, embeddings)

        # Define LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

        qa_yt = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=yt_docsearch.as_retriever()
        )

        # Get answer
        result = qa_yt.run(query)
        if isinstance(result, str):
            st.write("Answer:", result)
        else:
            st.write("Answer:", result.answer)
