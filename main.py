from dotenv import load_dotenv
import os
import warnings
import time
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Hide warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Load API keys
load_dotenv()

# Create DeepSeek LLM (streaming + medium length)
def get_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-chat-v3-0324:free",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.5,
        max_tokens=300,
        streaming=True
    )

llm = get_llm()

# Load and process document
loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Create embeddings + vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Function to ask questions with retries
def ask_question(query, retries=2):
    for attempt in range(retries):
        try:
            # Retrieve relevant context
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Combine into a prompt
            prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

            # Streaming response
            print("Bot:", end=" ", flush=True)
            for chunk in llm.stream(prompt):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print("\n")  # Newline after answer
            return
        except Exception as e:
            print(f"\n⚠️ Error: {e} (Attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                print("🔄 Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("❌ Failed to get a response after retries.")


# Interactive loop
print("\n💬 DeepSeek RAG Chatbot (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.strip().lower() in ["exit", "quit"]:
        print("Bot: Goodbye! 👋")
        break
    ask_question(query)
