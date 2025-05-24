import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from google.auth import default
from typing import List
import json

credentials, project_id = default()
if not credentials or not project_id:
    raise ValueError("Failed to obtain valid Google Cloud credentials")

# Initialize Vertex AI LLM


# Initialize the language model

# Define the output schema using Pydantic
class QuestionSet(BaseModel):
    questions: List[str] = Field(description="A list of four relevant sub-questions derived from the primary question and metadata.")

# Initialize the JSON output parser with the defined schema
parser = JsonOutputParser(pydantic_object=QuestionSet)

# Define the prompt template
prompt = PromptTemplate(
    template=(
        "Given the primary question: '{question}' and the following metadata columns: {metadata}, "
        "generate exactly four insightful and relevant sub-questions that delve deeper into the context of the primary question. "
        "Ensure the sub-questions are specific and can be addressed using the provided metadata.\n\n"
        "{format_instructions}"
    ),
    input_variables=["question", "metadata"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)



prompt_consolidated = PromptTemplate(
    template=(
        "You are provided with a series of questions and their corresponding answers:\n\n"
        "{qa_pairs}\n\n"
        "Based on the above, provide a comprehensive analysis that synthesizes the information, "
        "highlighting key insights, patterns, and any notable observations. "
        "Present the analysis in a clear and concise manner."
    ),
    input_variables=["qa_pairs"],
)


# Function to generate sub-questions
def generate_sub_questions(question: str, metadata: List[str], model):
    # Format the metadata as a comma-separated string
    metadata_str = ", ".join(metadata)
    # Create the prompt with the provided question and metadata
    formatted_prompt = prompt.format(question=question, metadata=metadata_str)
    # Invoke the language model with the formatted prompt
    response = model.invoke(formatted_prompt)
    # For VertexAI, the response is already a string, no need to access .content
    parsed_output = parser.parse(response)
    return parsed_output

def generate_consolidated_answer(qa_pairs: str, model):
    # Create the prompt with the provided QA pairs
    formatted_prompt = prompt_consolidated.format(qa_pairs=qa_pairs)
    # Invoke the language model with the formatted prompt
    response = model.invoke(formatted_prompt)
    # Return the response directly since we want a text analysis, not JSON
    return response

# Example usage

