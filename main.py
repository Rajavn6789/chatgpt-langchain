from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

# LangChain imports (Windows-safe)
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter  # or RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts_min.txt", encoding="utf-8")  # add encoding on Windows


docs = loader.load_and_split(text_splitter=text_splitter)

for doc in docs:
    print(doc.page_content)
    print("\n")
