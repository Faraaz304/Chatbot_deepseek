from dotenv import load_dotenv
import os
import warnings

# ✅ Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ✅ Load environment variables (make sure DEEPSEEK_API_KEY is in your .env file)
load_dotenv()

# ✅ Imports for DeepSeek LLM (OpenRouter)
from langchain_openai import ChatOpenAI

# ✅ RAG components
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize DeepSeek model
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model_name="deepseek/deepseek-chat-v3-0324:free",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Load the document
loader = TextLoader("data.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in Chroma (in-memory)
vectorstore = Chroma.from_documents(documents, embeddings)

# Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# ✅ Interactive chat loop
print("\n💬 DeepSeek RAG Chatbot (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.strip().lower() in ["exit", "quit"]:
        print("Bot: Goodbye! 👋")
        break
    answer = qa.run(query)
    print("Bot:", answer)
J