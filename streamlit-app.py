import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from google.auth import default
from dotenv import load_dotenv
import logging
import os
import gc
from typing import List

# Configure environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Disable file watcher for Streamlit to improve performance
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

class DataAnalyzer:
    """Class to manage data analysis using LangChain."""

    def __init__(self):
        """Initialize the data analyzer with Google Cloud authentication and LLM configuration."""
        try:
            # Get Google Cloud credentials
            credentials, project_id = default()
            if not credentials or not project_id:
                raise ValueError("Failed to obtain valid Google Cloud credentials")

            # Initialize Vertex AI LLM
            self.llm = VertexAI(
                model_name="gemini-1.5-pro",
                project=project_id,
                credentials=credentials,
                temperature=0.3, # Lower temperature for more deterministic sheet selection
                max_output_tokens=1000
            )

            # Initialize conversation memory
            self.memory = ConversationBufferMemory()

            logger.info(f"Data analyzer initialized successfully with project ID: {project_id}")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def analyze_data(self, dfs: dict, query):
        """
        Generate analysis of the dataframes based on the user query,
        allowing the agent to work with multiple dataframes.

        Args:
            dfs (dict): A dictionary where keys are sheet names and values are pandas.DataFrames.
            query (str): The user's question.

        Returns:
            str: The analysis result.
        """
        try:
            if not dfs:
                return "No data available for analysis."

            agent = create_pandas_dataframe_agent(
                self.llm,
                list(dfs.values()),
                verbose=True,
                memory=self.memory,
                allow_dangerous_code=True,
                return_intermediate_steps=True
            )

            prompt = f"""You have access to multiple pandas DataFrames, each corresponding to a sheet in an Excel file.
The sheet names are: {list(dfs.keys())}.

Use these DataFrames to identify the unique leave types taken by each employee.

Return a summary of each employee and the unique leave types they have taken."""

            response = agent(prompt)
            analysis_result = response['output']
            logger.info("Initial analysis by agent completed.")
            logger.info(f"Agent's response: {analysis_result}")

            # Post-process the data for a comprehensive list
            leave_data_df = dfs.get('LEAVE DATA-SEP')
            leave_types_df = dfs.get('Leave types')

            if leave_data_df is not None and leave_types_df is not None:
                merged_df = pd.merge(leave_data_df, leave_types_df, on='EMP_CODE', how='left')
                if not merged_df.empty:
                    employee_leave_types = merged_df.groupby('EMP_NAME')['LEAVE_TYPE'].unique()
                    output_list = [f"- **{emp}**: {types.tolist()}" for emp, types in employee_leave_types.items()]
                    final_output = "\n".join(output_list)
                    logger.info("Data post-processed to list all employees and unique leave types.")
                    return final_output
                else:
                    return "Could not merge the dataframes to find employee leave types."
            else:
                return "One or both of the 'LEAVE DATA-SEP' or 'Leave types' sheets were not loaded correctly."

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while analyzing your data: {str(e)}"

def load_excel_metadata(uploaded_file):
    """
    Loads metadata (sheet names and column names) from an Excel file.

    Args:
        uploaded_file: The uploaded file object.

    Returns:
        dict: A dictionary where keys are sheet names and values are lists of column names.
              Returns None if there's an error reading the file.
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        metadata = {}
        for sheet_name in sheet_names:
            df_head = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=1)
            metadata[sheet_name] = df_head.columns.tolist()
        return metadata
    except Exception as e:
        logger.error(f"Error loading Excel metadata: {str(e)}")
        return None

def load_excel_data(uploaded_file, sheet_name, max_rows=100000):
    """
    Loads data from a specific sheet of an Excel file.

    Args:
        uploaded_file: The uploaded file object.
        sheet_name (str): The name of the sheet to load.
        max_rows (int): Maximum number of rows to load.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame, or None if loading fails.
    """
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=max_rows)
        if len(df) == max_rows:
            st.warning(f"Only loaded first {max_rows} rows from sheet '{sheet_name}'.")
        logger.info(f"Successfully loaded sheet '{sheet_name}' with shape: {df.shape}")
        # Convert problematic data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.isna().sum() < 0.5 * len(df):
                        df[col] = numeric_col
                except:
                    pass
        return df
    except Exception as e:
        logger.error(f"Error loading sheet '{sheet_name}': {str(e)}")
        return None
    finally:
        gc.collect()

def get_relevant_sheets(query: str, excel_metadata: dict, llm):
    """
    Determines which sheets from the Excel file are relevant to answer the user's query,
    considering both sheet names and column names.

    Args:
        query (str): The user's question.
        excel_metadata (dict): A dictionary where keys are sheet names and values
                               are lists of column names for that sheet.
        llm: The language model.

    Returns:
        List[str]: A list of sheet names that are likely relevant to the query.
    """
    prompt = PromptTemplate(
        template=(
            "Given the following user query: '{query}'\n"
            "And the following information about the sheets in an Excel file:\n"
            "{sheet_info}\n"
            "Which of these sheets are most likely to contain information relevant to the query? "
            "Please list only the names of the relevant sheets, separated by commas. "
            "If none are relevant, just say 'None'."
        ),
        input_variables=["query", "sheet_info"],
    )

    sheet_info_str = ""
    for sheet_name, columns in excel_metadata.items():
        sheet_info_str += f"- Sheet Name: '{sheet_name}', Columns: {columns}\n"

    formatted_prompt = prompt.format(query=query, sheet_info=sheet_info_str)
    response = llm.invoke(formatted_prompt)
    relevant_sheets_str = response.strip()
    if relevant_sheets_str.lower() == "none":
        return []
    else:
        return [sheet.strip() for sheet in relevant_sheets_str.split(',')]

def initialize_session_state():
    """Initialize all session state variables needed for the app."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'df' not in st.session_state:
        st.session_state.df = None

    if 'file_name' not in st.session_state:
        st.session_state.file_name = None

    if 'excel_metadata' not in st.session_state:
        st.session_state.excel_metadata = None

    if 'sheet_names' not in st.session_state:
        st.session_state.sheet_names = None

    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

def clear_chat_history():
    st.session_state.chat_history = []

def main():
    """Main function to run the Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="Chat with your data",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # App title and description
    st.title("📊 Chat with your data")
    st.markdown("""
    Upload an Excel file and ask questions about your data in natural language.
    The AI will analyze the relevant sheets and provide insights.
    """)

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Excel File")
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

        if uploaded_file is not None:
            logger.info("Excel file uploaded.")
            try:
                # Get sheet and column metadata
                metadata = load_excel_metadata(uploaded_file)
                if metadata:
                    st.session_state.excel_metadata = metadata
                    st.session_state.sheet_names = list(metadata.keys())
                    logger.info(f"Excel metadata loaded. Sheets: {st.session_state.sheet_names}")
                else:
                    st.error("Could not read Excel metadata.")
                    logger.error("Could not read Excel metadata.")
                    return

                # Initialize analyzer if not already done
                if st.session_state.analyzer is None:
                    st.session_state.analyzer = DataAnalyzer()
                    logger.info("DataAnalyzer initialized.")

            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                logger.error(f"Error processing uploaded file: {e}")

        # Add additional options
        if st.session_state.sheet_names:
            st.header("Options")
            if st.button("Clear Chat History"):
                clear_chat_history()
            if st.session_state.chat_history and st.button("Export Chat"):
                chat_text = "\n\n".join([f"**{role.upper()}**: {message}" for role, message in st.session_state.chat_history])
                st.download_button(
                    label="Download Chat",
                    data=chat_text,
                    file_name="excel_analyzer_chat.md",
                    mime="text/markdown"
                )

        st.header("About")
        st.info("""
        This app analyzes Excel data using natural language queries.
        Powered by LangChain and Vertex AI.
        """)

    # Main area with chat interface
    if "excel_metadata" in st.session_state and st.session_state.excel_metadata:
        st.header("Chat with Your Data")

        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        # Input area for new queries
        if prompt := st.chat_input("Ask a question about your data"):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                relevant_sheets = get_relevant_sheets(
                    prompt,
                    st.session_state.excel_metadata,
                    st.session_state.analyzer.llm
                )
                st.info(f"LLM thinks these sheets are relevant: {', '.join(relevant_sheets)}")

                dfs = {}
                if relevant_sheets:
                    with st.spinner(f"Loading relevant sheets..."):
                        all_loaded = True
                        for sheet in relevant_sheets:
                            df = load_excel_data(uploaded_file, sheet_name=sheet, max_rows=100000)
                            if df is not None and not df.empty:
                                dfs[sheet] = df
                            elif df is None:
                                all_loaded = False
                                st.error(f"Failed to load sheet: {sheet}")
                        if dfs:
                            analysis_result = st.session_state.analyzer.analyze_data(dfs, prompt) # Get the post-processed result
                            st.session_state.chat_history.append(("ai", analysis_result))
                            with st.chat_message("ai"):
                                st.markdown(analysis_result)
                        elif all_loaded:
                            st.warning("No relevant data found in the loaded sheets for your query.")
                else:
                    st.warning("Could not determine relevant sheets. Please try again.")
    else:
        st.info("👈 Please upload an Excel file from the sidebar to get started!")

if __name__ == "__main__":
    main()