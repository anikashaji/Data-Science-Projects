import os
from autogen import AssistantAgent
from dotenv import load_dotenv
from vertexai.generative_models import HarmBlockThreshold, HarmCategory
import logging
import google.auth

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ResearchAgents:
   def __init__(self):
       try:
           credentials, project_id = google.auth.default()
       except Exception as e:
           logger.error(f"Authentication failed: {e}")
           raise

       safety_settings = {
           HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
           HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
           HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
           HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
       }

       # Commented out Groq configuration
       # self.llm_config = {'config_list': [{'model': 'llama-3.3-70b-versatile', 'api_key': self.groq_api_key, 'api_type': "groq"}]}
       
       self.llm_config = {
           'config_list': [{
               "model": "gemini-1.5-pro",
               "api_type": "google",
               "project_id": project_id,
               "credentials": credentials,
               "location": "us-west1"
           }],
           "temperature": 0.7,
           "max_tokens": 2000
       }
       
       self.analyzer = AssistantAgent(
           name="Analyzer",
           system_message="analyze and sumarize the data by pointing out valid data points. "
                         " generate a detailed summary .",
           llm_config=self.llm_config,
           human_input_mode="NEVER",
           code_execution_config=False
       )

   def Analyze_docs(self, df,query):
       """Generates a summary of the research paper."""
       analysis_prompt = f"""
           YOU ARE AN ADVANCED SYSTEM DESIGNED TO ANSWER QUESTIONS FROM STRUCTURED DATA WITH DEPTH AND NUANCE.

           FOLLOWING IS THE DATAFRAME YOU NEED TO ANALYZE:
           {df}

           PLEASE ANSWER THE FOLLOWING QUESTION:
           {query}
           IMPORTANT: Analyze this with depth.
           """
       summary_response = self.analyzer.generate_reply(
           messages=[{"role": "user", "content": analysis_prompt}]
       )
       return summary_response.get("content", "Summarization failed!") if isinstance(summary_response, dict) else str(summary_response)