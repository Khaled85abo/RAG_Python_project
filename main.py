from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from clear_screen import clear_screen
from llama3_lm import llama3_lm

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

EMBEDDINGS_SIZE_LARGE = 3072
EMBEDDINGS_SIZE_SMALL = 1536

def get_openai_embedding_model():
    try:
        return OpenAIEmbeddings( model ="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    except:
        raise ConnectionError("Error loading OpenAI embeddings")


def get_faiss_vector_store(path, embeddings):
    return FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization= True)

def search_similarities(library, query):
    try:
        return library.similarity_search(query, 5)
    except:
        raise ConnectionError("Error searching similarities")

if __name__ == "__main__":
    local_VDB_path = "./python_code_library_FAISS"
    embeddings = get_openai_embedding_model()
    library = get_faiss_vector_store(path=local_VDB_path, embeddings=embeddings)
    print("main is running")
    while True:
        clear_screen()
        query = input("Enter a query: ")
        query_answer = search_similarities(library, query)
        # print(f"Source: {query_answer[0].metadata['source']}")
        # print(f"Type: {query_answer[0].metadata['type']}")
        # print(f"Name: {query_answer[0].metadata['name']}")
        # print(f"Page content: {query_answer[0].page_content}")
        # print("------------------------------------------")
        for answer in range(3):
            print(f"Source: {query_answer[answer].metadata['source']}")
            print(f"Type: {query_answer[answer].metadata['type']}")
            print(f"Name: {query_answer[answer].metadata['name']}")
            print(f"Page content: {query_answer[answer].page_content}")
            print("------------------------------------------")
        llama3_answer = llama3_lm(f"{query} {query_answer}")
        print(f"answer from llama3: {llama3_answer}")
        input("Press Enter to continue...")
