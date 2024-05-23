from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import ast
import astor
import openai
from clear_screen import clear_screen

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

EMBEDDINGS_SIZE_LARGE = 3072
EMBEDDINGS_SIZE_SMALL = 1536

embeddings = OpenAIEmbeddings( model ="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)

def get_openai_embedding_model():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings()

def get_faiss_vector_store():
    return FAISS.load_local("./python_code_library_FAISS", embeddings=embeddings, allow_dangerous_deserialization= True)

def search_similarities(library, query):
    return library.similarity_search(query, 5)

if __name__ == "__main__":
    library = get_faiss_vector_store()
    print("main is running")
    while True:
        clear_screen()
        query = input("Enter a query: ")
        query_answer = search_similarities(library, query)
        print(f"Source: {query_answer[0].metadata['source']}")
        print(f"Type: {query_answer[0].metadata['type']}")
        print(f"Name: {query_answer[0].metadata['name']}")
        print(f"Page content: {query_answer[0].page_content}")
        input("Press Enter to continue...")