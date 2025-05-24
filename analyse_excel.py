import pandas as pd
from llama_index.core import Settings
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.vertex import Vertex
from google.auth import default
from agents import ResearchAgents

# Initialize credentials and models
credentials, project_id = default()

# Set up the LLM with Gemini
llm = Vertex(
    model="gemini-1.5-pro-002",
    project=project_id,
    credentials=credentials
)

# Configure global settings
Settings.llm = llm

def analyze_excel(excel_path, sheet_name="LEAVE DATA-SEP", natural_language_query=None):
    """
    Analyze Excel file using natural language query
    
    Args:
        excel_path (str): Path to the Excel file
        sheet_name (str, optional): Name of the sheet to analyze. If None, uses the first sheet.
        natural_language_query (str, optional): The query in natural language
        
    Returns:
        Response from the query engine
    """
    # Read Excel file
    if sheet_name:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(excel_path)
    
    print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create Pandas query engine
    query_engine = PandasQueryEngine(
        df=df,
        verbose=True,
        llm=llm
    )
    return df
    


# Example usage
if __name__ == "__main__":
    # Path to your Excel file
    excel_file_path = "LEAVE DATA SEP2024.xlsx"
    
    # Sample queries for customer data analysis
    sample_queries = [
        "analize the whole leave data of employees and summaize it without losing its context"
    ]
    
    print("Excel File Analysis Results:")
    print("-" * 50)
    
    # First, show basic dataframe information without a specific query
    print("\nBasic DataFrame Analysis:")
    basic_analysis = analyze_excel(excel_file_path)
    print(f"Result: {basic_analysis}")
    print("-" * 30)
    
    generator = ResearchAgents()
    
    # Then run sample queries
    for query in sample_queries:
        print(f"\nQuery: {query}")
        result = analyze_excel(excel_file_path, natural_language_query=query)
        analysis_result = generator.Analyze_docs(result)
        print(f"Result: {analysis_result}")
        print("-" * 30)