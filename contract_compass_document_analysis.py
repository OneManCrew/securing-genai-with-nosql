import os
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizer
from transformers.pipelines import Pipeline

async def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()

async def create_mongo_client() -> MongoClient:
    """Create and return a MongoDB client using environment variables for configuration."""
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        raise ValueError("MONGO_URI is not set in the environment variables")
    return MongoClient(mongo_uri)

async def setup_nlp_pipeline() -> Pipeline:
    """Setup and return a Hugging Face Transformer pipeline for NLP tasks."""
    return pipeline("text-classification", model='distilbert-base-uncased')

async def analyze_documents(client: MongoClient, nlp_pipeline: Pipeline) -> List[Dict[str, str]]:
    """Analyze legal documents stored in MongoDB using a Transformer-based NLP pipeline."""
    db = client['contract_db']
    collection = db['contracts']
    results = []

    async for document in collection.find({}):
        text = document.get('content', '')
        if text:
            analysis_result = nlp_pipeline(text)
            results.append({
                'contract_id': document['_id'],
                'analysis': analysis_result
            })
    return results

async def main() -> None:
    """Main function to coordinate the document analysis process."""
    await load_environment()
    client = await create_mongo_client()
    nlp_pipeline = await setup_nlp_pipeline()
    results = await analyze_documents(client, nlp_pipeline)

    # Output results
    for result in results:
        print(f"Contract ID: {result['contract_id']}, Analysis: {result['analysis']}")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())