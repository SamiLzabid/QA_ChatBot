import streamlit as st
import os
import tempfile
from openai import OpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma


if "OPENROUTER_API_KEY" in st.secrets and "GROQ_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("🔑 Missing API Keys. Ensure both OPENROUTER_API_KEY and GROQ_API_KEY are configured in your secrets.toml.")
    st.stop()

class OpenRouterEmbeddings(Embeddings):
    """
    Custom LangChain Embedding wrapper designed to map text document strings 
    into OpenRouter's specific nested object payload structure.
    """
    def __init__(self, model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        # Process chunks in batches of 16 to respect upstream size limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Format text inputs into the structural content array layout
            formatted_input = [{"content": [{"type": "text", "text": t}]} for t in batch_texts]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=formatted_input,
                encoding_format="float"
            )
            embeddings.extend([data.embedding for data in response.data])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        formatted_input = [{"content": [{"type": "text", "text": text}]}]
        response = self.client.embeddings.create(
            model=self.model,
            input=formatted_input,
            encoding_format="float"
        )
        return response.data[0].embedding


# 1. Loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        st.sidebar.text(f'Loading {os.path.basename(file)}...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        st.sidebar.text(f'Loading {os.path.basename(file)}...')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        st.sidebar.text(f'Loading {os.path.basename(file)}...')
        loader = TextLoader(file)
    else:
        st.sidebar.error('Document format is not supported!')
        return None

    data = loader.load()
    return data

# 2. Wikipedia Loader
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    st.sidebar.text(f"Searching Wikipedia for '{query}'...")
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data
  
# 3. Chunking Data
def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

# 4. Creating Vector Database
def create_embeddings_chroma(chunks):
    embeddings = OpenRouterEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)  # in-memory, no file lock
    return vector_store

# 5. Loading existing Vector Database
def load_embeddings_chroma(persist_directory='./chroma_db'):
    embeddings = OpenRouterEmbeddings() 
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 
    return vector_store

# 6. Initializing Conversational Chain pointed entirely to OpenRouter Endpoints
def build_conversational_chain(vector_store):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    # Reformulates vague or follow-up questions into standalone searchable queries
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are reformulating user questions for a document retrieval system. "
         "Given the chat history and the latest user question, rewrite it as a clear, "
         "standalone question that can be searched against the document. "
         "If the user asks something like 'what is this document about' or 'summarize the document', "
         "rewrite it as 'What are the main topics and contents of this document?'. "
         "Do NOT answer the question — only rewrite it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Answers using retrieved document chunks — concise by instruction
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a document the user has uploaded. "
         "The text chunks below are extracted directly from that document — treat them as the user's document. "
         "Use only the provided context to answer. "
         "Keep answers concise: 2-3 sentences for simple questions, short paragraphs only when detail is truly needed. "
         "If the context does not contain enough information to answer, say so briefly — do not fabricate details.\n\n"
         "Document context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


def ask_question(q, chain, chat_history):
    result = chain.invoke({
        "input": q,
        "chat_history": chat_history   
    })
    return result.get("answer", "No response generated.")


st.set_page_config(page_title="OpenRouter RAG Chatbot", layout="wide")
st.title("🌐 Fully OpenRouter-Powered AI Document Assistant")

# Initialize persistent session configurations
if "messages" not in st.session_state:
    st.session_state.messages = []
if "crc" not in st.session_state:
    st.session_state.crc = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  

# Sidebar layout
st.sidebar.header("Configuration Panel")
st.sidebar.markdown("---")
st.sidebar.subheader("Knowledge Ingestion")
data_source = st.sidebar.radio("Choose Source Type:", ("Upload Documents", "Wikipedia Fetch"))

data = None

if data_source == "Upload Documents":
    uploaded_file = st.sidebar.file_uploader("Select PDF, DOCX, or TXT file:", type=['pdf', 'docx', 'txt'])

    if uploaded_file is not None:
        if st.sidebar.button("Process & Embed File"):
            file_extension = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                with st.spinner("Processing document segments using OpenRouter Embeddings..."):
                    data = load_document(temp_file_path)
                    if data:
                        chunks = chunk_data(data, chunk_size=256)
                        vector_store = create_embeddings_chroma(chunks)
                        st.session_state.crc = build_conversational_chain(vector_store)
                        st.session_state.chat_history = []  
                        st.session_state.messages = []
                        st.sidebar.success("Database compiled successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing file components: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

elif data_source == "Wikipedia Fetch":
    wiki_query = st.sidebar.text_input("Wikipedia Topic Search Context:")
    max_docs = st.sidebar.number_input("Max Articles to pull:", min_value=1, max_value=5, value=2)

    if st.sidebar.button("Fetch & Embed Wiki Content"):
        if not wiki_query.strip():
            st.sidebar.error("Please provide a search context query.")
        else:
            try:
                with st.spinner("Fetching and indexing Wikipedia datasets..."):
                    data = load_from_wikipedia(query=wiki_query, load_max_docs=int(max_docs))
                    if data:
                        chunks = chunk_data(data, chunk_size=256)
                        vector_store = create_embeddings_chroma(chunks)
                        st.session_state.crc = build_conversational_chain(vector_store)
                        st.session_state.chat_history = []  # Reset history on new source
                        st.session_state.messages = []
                        st.sidebar.success("Wikipedia content indexed completely!")
            except Exception as e:
                st.sidebar.error(f"Error collecting wiki contents: {e}")

if st.sidebar.button("Clear Conversation History"):
    st.session_state.messages = []
    st.session_state.chat_history = []  
    st.rerun()


if st.session_state.crc is None:
    st.info("Ingest data or query a Wikipedia topic on the sidebar layout to initialize your system pipeline.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ask a question regarding your uploaded knowledge documents:"):

        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("assistant"):
            with st.spinner("OpenRouter LLM is thinking..."):
                try:
                    answer = ask_question(
                        user_prompt,
                        st.session_state.crc,
                        st.session_state.chat_history  
                    )
                    
                    st.session_state.chat_history.extend([
                        HumanMessage(content=user_prompt),
                        AIMessage(content=answer)
                    ])

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Inference Connection Error: {e}")