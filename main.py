from dotenv import load_dotenv
load_dotenv(override=True)  


from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter  # or RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts_min.txt", encoding="utf-8") 


docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

for result in results:
    print("\n")
    print(result.page_content)
