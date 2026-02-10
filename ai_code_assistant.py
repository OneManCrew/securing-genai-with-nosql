import os
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pymongo
from pymongo.errors import ConnectionFailure

class CodeAssistant:
    def __init__(self, model_name: str, db_uri: str):
        """
        Initializes the AI-driven code assistant with a specified model and database connection.

        :param model_name: Name of the Transformer model to use for code analysis.
        :param db_uri: URI for connecting to the NoSQL database.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.nlp_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        
        # Establishing a connection to the NoSQL database
        self.client = pymongo.MongoClient(db_uri)
        try:
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
        except ConnectionFailure:
            raise RuntimeError("Server not available")
        self.db = self.client['code_analysis']

    async def analyze_code_snippet(self, code_snippet: str) -> Dict[str, Any]:
        """
        Analyzes a given code snippet using AI to provide insights and stores them in a NoSQL database.

        :param code_snippet: A string containing the code snippet to analyze.
        :return: A dictionary with the analysis results.
        """
        # Using the AI model to generate insights
        insights = self.nlp_pipeline(code_snippet, max_length=512, truncation=True)
        
        # Store insights in the NoSQL database
        result = self.db.insights.insert_one({
            "code_snippet": code_snippet,
            "insights": insights
        })
        
        return {
            "code_snippet": code_snippet,
            "insights": insights,
            "db_record_id": str(result.inserted_id)
        }

# Example usage
async def main():
    db_uri = os.getenv("MONGO_DB_URI")  # Ensure MONGO_DB_URI is set in the environment
    assistant = CodeAssistant(model_name="t5-base", db_uri=db_uri)
    code_snippet = "def add(a, b): return a + b"
    analysis_result = await assistant.analyze_code_snippet(code_snippet)
    print(analysis_result)

# To execute the main function when running the script
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())