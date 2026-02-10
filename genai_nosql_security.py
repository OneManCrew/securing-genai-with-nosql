import os
from typing import Any, Dict
import logging
from langchain import OpenAI, LangChain
from langchain.chains import RetrievalAugmentedGenerationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import NoSQLVectorStore
from langchain.prompts import PromptTemplate
from pymongo import MongoClient

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)

class GenAINoSQLSecurity:
    """
    This class demonstrates the integration of Generative AI with NoSQL databases,
    using LangChain for retrieval-augmented generation and a MongoDB backend.
    """
    def __init__(self, mongo_uri: str, openai_api_key: str) -> None:
        self.client = MongoClient(mongo_uri)
        self.db = self.client['genai_security_db']
        self.collection = self.db['documents']
        self.openai_api_key = openai_api_key
        self.llm = OpenAI(api_key=openai_api_key)
        self.embedding = OpenAIEmbeddings(api_key=openai_api_key)
        self.vector_store = NoSQLVectorStore(self.db, self.embedding)
        self.chain = RetrievalAugmentedGenerationChain(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            prompt_template=PromptTemplate("Use context to answer the question: {question}")
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the system with a natural language question, utilizing RAG pattern for optimal response.
        """
        logging.info(f"Querying with question: {question}")
        try:
            result = self.chain.run(question=question)
            logging.info(f"Query result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error during query: {e}")
            return {"error": str(e)}

# Configuration and environment handling
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
openai_api_key = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')

# Example usage
if __name__ == "__main__":
    genai_security = GenAINoSQLSecurity(mongo_uri=mongo_uri, openai_api_key=openai_api_key)
    response = genai_security.query("What are the security considerations when using NoSQL databases?")
    print(response)