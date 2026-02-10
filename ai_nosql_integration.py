import os
from typing import Dict, Any
import asyncio
import uvloop
from pymongo import MongoClient, errors
from openai import OpenAI
from openai.embeddings_utils import get_embedding

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

class AIOpenAIHandler:
    """
    Asynchronous handler for interfacing with OpenAI's API, specifically to leverage the embedding capabilities
    for processing AI tasks and integrating with MongoDB for storage and retrieval.
    """

    def __init__(self, api_key: str, mongo_uri: str, db_name: str):
        self.openai = OpenAI(api_key=api_key)
        self.client = MongoClient(mongo_uri)
        self.db_name = db_name

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process the text to generate embeddings and store/retrieve data from a NoSQL database.

        :param text: The text input to be processed
        :return: The document retrieved from the database
        """
        try:
            # Generate embedding asynchronously
            embedding = await asyncio.to_thread(get_embedding, text)

            # Connect to the database
            db = self.client[self.db_name]
            collection = db['text_embeddings']

            # Store the embedding
            collection.insert_one({'text': text, 'embedding': embedding})

            # Retrieve document for demonstration purposes
            document = collection.find_one({'text': text})

            return document

        except errors.PyMongoError as e:
            print(f"MongoDB Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

async def main():
    handler = AIOpenAIHandler(api_key=os.getenv('OPENAI_API_KEY'),
                              mongo_uri=os.getenv('MONGO_URI'),
                              db_name='ai_operations')
    text = "Integrating AI models with NoSQL databases"
    result = await handler.process_text(text)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())