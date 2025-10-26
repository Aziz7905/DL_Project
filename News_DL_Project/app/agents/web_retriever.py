# agents/web_retriever.py
import os
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

load_dotenv()

class WebRetrieverAgent:
    def __init__(self):
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("SerpAPI key not found. Please set SERPAPI_API_KEY in your environment or .env file.")
        self.search = SerpAPIWrapper(serpapi_api_key=api_key)

    def get_live_evidence(self, claim):
        # Return top web results as a list of dicts with snippet/link
        results = self.search.results(claim)  # Use results() not run()
        if "organic_results" in results:
            return results["organic_results"]  # list of dict
        else:
            return []
