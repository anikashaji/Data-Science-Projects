import psycopg2
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
from typing import List, Tuple
from psycopg2 import sql
import time # Import time

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
                temperature=0.7,
                max_output_tokens=2000
            )

            # Initialize conversation memory
            self.memory = ConversationBufferMemory()

            logger.info(f"Data analyzer initialized successfully with project ID: {project_id}")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def analyze_data(self, df, query):
        """
        Generate analysis of the dataframe based on the user query.

        Args:
            df (pandas.DataFrame): The dataframe to analyze
            query (str): The user's question

        Returns:
            str: The analysis result
        """
        try:
            if df is None or df.empty:
                return "No data available for analysis. Please upload a valid Excel file."

            # Create a pandas agent for data analysis with code execution enabled
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                memory=self.memory,
                allow_dangerous_code=True  # Enable code execution for data analysis
            )

            # Generate response
            response = agent.run(query)
            logger.info("Analysis completed successfully")
            return response

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while analyzing your data: {str(e)}"

    def analyze_data_stream(self, dfs: dict, query: str):
        """
        Generate a streaming analysis of the dataframes based on the user query.

        Args:
            dfs (dict): A dictionary of pandas.DataFrame to analyze, where keys are sheet/table names.
            query (str): The user's question.

        Yields:
            str: A chunk of the analysis result.
        """
        try:
            if not dfs:
                yield "No data available for analysis. Please upload a valid Excel file or connect to the database."
                return

            combined_analysis = ""
            for name, df in dfs.items():
                if df is None or df.empty:
                    yield f"No data in '{name}'.\n"
                    continue

                # Create a pandas agent for data analysis with code execution enabled
                agent = create_pandas_dataframe_agent(
                    self.llm,
                    df,
                    verbose=True,
                    memory=self.memory,
                    allow_dangerous_code=True,   # Enable code execution for data analysis
                    return_intermediate_steps=True
                )

                # Generate response
                logger.info(f"Analyzing data from: '{name}' with query: '{query}' (streaming)")
                response = agent(query) # Note: Pandas agent doesn't inherently stream.

                # For non-streaming agent, we just yield the final output.
                full_response = f"**Analysis from '{name}':**\n{response['output']}\n\n"
                for word in full_response.split():
                    yield word + " "
                    time.sleep(0.01) # Simulate streaming

            logger.info("Streaming analysis attempted.")

        except Exception as e:
            error_msg = f"Error during streaming analysis: {str(e)}"
            logger.error(error_msg)
            yield f"I encountered an error while analyzing your data: {str(e)}"

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

def load_database_tables(conn, table_names: List[str], max_rows: int) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    """Loads data from specified PostgreSQL tables into pandas DataFrames and returns messages."""
    loaded_dfs = []
    successfully_loaded_names = []
    messages = []
    if conn:
        cursor = conn.cursor()
        messages.append(f"Connected to PostgreSQL database: `{conn.get_dsn_parameters()['dbname']}`")
        for table_name in table_names:
            try:
                query = sql.SQL("SELECT * FROM {} LIMIT %s;").format(sql.Identifier(table_name))
                cursor.execute(query, (max_rows,))
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                loaded_dfs.append(df)
                successfully_loaded_names.append(table_name)
            except psycopg2.Error as e:
                messages.append(f"Error loading table '{table_name}': {e}")
        cursor.close()
    return loaded_dfs, successfully_loaded_names, messages

def load_single_excel_sheet(uploaded_file, selected_sheet, max_rows):
    """Loads a single sheet from an Excel file into a DataFrame."""
    df = load_excel_data(uploaded_file, selected_sheet, max_rows)
    return {selected_sheet: df} if df is not None else None, [selected_sheet] if df is not None else []

def initialize_session_state():
    """Initialize all session state variables needed for the app."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None

    if 'file_name' not in st.session_state:
        st.session_state.file_name = None

    if 'excel_metadata' not in st.session_state:
        st.session_state.excel_metadata = None

    if 'sheet_names' not in st.session_state:
        st.session_state.sheet_names = None

    if 'loaded_sheet_names' not in st.session_state:
        st.session_state.loaded_sheet_names = None

    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = None

    if 'loaded_dfs' not in st.session_state:
        st.session_state.loaded_dfs = []

    if 'loaded_table_names' not in st.session_state:
        st.session_state.loaded_table_names= []

    if 'loading_messages' not in st.session_state:
        st.session_state.loading_messages = []

    if 'is_multisheet' not in st.session_state:
        st.session_state.is_multisheet = False

def clear_chat_history():
    st.session_state.chat_history = []

def initialize_analyzer():
    """Initializes the DataAnalyzer if it's not already initialized."""
    if st.session_state.analyzer is None:
        try:
            st.session_state.analyzer = DataAnalyzer()
            logger.info("DataAnalyzer initialized.")
        except Exception as e:
            st.error(f"Failed to initialize the AI analyzer: {e}")
            logger.error(f"Failed to initialize the AI analyzer: {e}", exc_info=True)

def clear_analyzer_memory():
    """Clears the memory of the DataAnalyzer."""
    if st.session_state.analyzer:
        st.session_state.analyzer.memory.clear()
        logger.info("Analyzer memory cleared.")

def load_excel_sheets(uploaded_file, sheet_names: List[str], max_rows: int) -> Tuple[dict, List[str]]:
    """Loads multiple sheets from an Excel file into a dictionary of DataFrames."""
    loaded_dfs = {}
    successfully_loaded = []
    for sheet_name in sheet_names:
        df = load_excel_data(uploaded_file, sheet_name, max_rows)
        if df is not None and not df.empty:
            loaded_dfs[sheet_name] = df
            successfully_loaded.append(sheet_name)
    return loaded_dfs, successfully_loaded

def handle_query_submit(prompt):
    """Handle submission of a query in the chat interface."""
    if not prompt:
        st.warning("Please enter a question.")
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    actual_query_for_agent = prompt
    if st.session_state.is_multisheet and isinstance(st.session_state.loaded_data, dict) and len(st.session_state.loaded_data) > 1:
        # Ensure loaded_sheet_names are strings for join
        sheet_names_str = ", ".join([str(name) for name in st.session_state.loaded_sheet_names])
        multi_sheet_context_hint = (
            f"Context: The data is spread across multiple tables (Excel sheets) named: [{sheet_names_str}]. "
            "You have access to all these tables. "
            "When a question is asked, determine if the necessary information might be in one or more of these tables based on their column names and the query. "
            "If information from multiple tables is needed, try to combine or relate them appropriately to answer the question. "
            "When referring to data, if possible, mention which sheet or table the information came from if it adds clarity.\n"
        )
        actual_query_for_agent = f"{multi_sheet_context_hint}User Question: {prompt}"
    elif isinstance(st.session_state.loaded_data, dict) and len(st.session_state.loaded_data) == 1:
        # If only one sheet is loaded, no need for multi-sheet context hint
        pass

    logger.info(f"Final query for agent: {actual_query_for_agent}")

    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response_content = []
        try:
            if isinstance(st.session_state.loaded_data, dict):
                for chunk_from_analyzer in st.session_state.analyzer.analyze_data_stream(st.session_state.loaded_data, actual_query_for_agent):
                    if chunk_from_analyzer:
                        words = str(chunk_from_analyzer).split(" ") # Ensure chunk is string
                        for i, word in enumerate(words):
                            full_response_content.append(word)
                            if i < len(words) - 1:
                                full_response_content.append(" ")
                            response_placeholder.markdown("".join(full_response_content) + "▌")
                            time.sleep(0.01) # Adjusted speed
                    time.sleep(0.01)
            else:
                st.error("No data loaded for analysis.")
                full_response_content.append("Error: No data loaded.")
        except Exception as stream_e:
            logger.error(f"Error during response streaming: {stream_e}", exc_info=True)
            full_response_content.append(f"\n\nAn error occurred while generating the full response: {stream_e}")
        response_placeholder.markdown("".join(full_response_content))
        st.session_state.chat_history.append({"role": "ai", "content": "".join(full_response_content)})

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
    Upload an Excel file or connect to a PostgreSQL database and ask questions about your data in natural language.
    The AI will analyze the relevant data and provide insights.
    """)

    # Sidebar for data source selection
    with st.sidebar:
        st.header("Choose Data Source")
        data_source = st.radio("Select data source:", ["Excel File", "PostgreSQL Database"])

        if data_source == "Excel File":
            st.header("1. Upload Excel File")
            uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    xls_inspect = pd.ExcelFile(uploaded_file)
                    all_sheet_names = xls_inspect.sheet_names

                    st.header("2. Select Sheet(s) & Load")
                    selected_sheet_names = st.multiselect(
                        "Which sheet(s) would you like to analyze?",
                        all_sheet_names,
                        default=all_sheet_names[0] if all_sheet_names else []
                    )

                    max_rows_per_sheet = st.number_input("Maximum rows to load per sheet",
                                                        min_value=100,
                                                        max_value=200000,
                                                        value=50000,
                                                        help="Limits rows per sheet for performance and memory.")

                    if st.button("Load Data into AI", key="load_excel_button"):
                        if not selected_sheet_names:
                            st.warning("Please select at least one sheet to load.")
                        else:
                            with st.spinner(f"Loading and processing {len(selected_sheet_names)} sheet(s)... Please wait."):
                                loaded_dfs, successfully_loaded_names = load_excel_sheets(uploaded_file, selected_sheet_names, max_rows_per_sheet)
                                if loaded_dfs:
                                    st.session_state.loaded_data = loaded_dfs
                                    st.session_state.loaded_sheet_names = successfully_loaded_names
                                    st.session_state.file_name = uploaded_file.name
                                    st.session_state.is_multisheet = len(loaded_dfs) > 1
                                    initialize_analyzer()
                                    if st.session_state.analyzer:
                                        st.success(f"Great! I've loaded sheet(s): **{', '.join(successfully_loaded_names)}** from **{uploaded_file.name}**. You can now start chatting!")
                                        st.session_state.chat_history = []
                                        clear_analyzer_memory()
                                        # Load metadata after successful load
                                        uploaded_file.seek(0)
                                        st.session_state.excel_metadata = load_excel_metadata(uploaded_file)
                                    else:
                                        st.error("Failed to initialize the AI analyzer. Please check Google Cloud setup.")
                                else:
                                    st.error("Could not load data from the selected sheets. Please check the file and your selections.")
                except Exception as e:
                    st.error(f"Error reading Excel file structure: {str(e)}. Ensure it's a valid .xlsx or .xls file.")
                    logger.error(f"Error processing uploaded file: {e}", exc_info=True)

        elif data_source == "PostgreSQL Database":
            st.header("1. Connect to PostgreSQL")
            db_host = os.getenv("POSTGRES_HOST")
            db_name = os.getenv("POSTGRES_DBNAME")
            db_user = os.getenv("POSTGRES_USER")
            db_password = os.getenv("POSTGRES_PASSWORD")

            if st.button("Connect to Database", key="connect_db_button"):
                if not all([db_host, db_name, db_user, db_password]):
                    st.error("Please fill in all database connection fields.")
                else:
                    try:
                        st.info(f"Host: {db_host}, Database: {db_name}, User: {db_user}, Password: {db_password}")
                        conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_password)
                        st.session_state.db_connection = conn
                        cursor = conn.cursor()
                        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                        table_names = [row[0] for row in cursor.fetchall()]
                        st.session_state.table_names = table_names
                        st.success(f"Successfully connected to database: `{db_name}`.")
                        cursor.close()
                    except psycopg2.Error as e:
                        st.error(f"Could not connect to the database: {e}")
                        st.session_state.db_connection = None
                        if 'table_names' in st.session_state:
                            del st.session_state.table_names

            if "table_names" in st.session_state:
                st.header("2. Select Table(s) & Load")
                selected_tables = st.multiselect(
                    "Which table(s) would you like to analyze?",
                    st.session_state.table_names,
                    default=st.session_state.table_names[0] if st.session_state.table_names else []
                )

                max_rows_per_table = st.number_input("Maximum rows to load per table",
                                                     min_value=100,
                                                     max_value=200000,
                                                     value=50000,
                                                     help="Limits rows per table for performance and memory.")

                if st.button("Load Data from DB", key="load_db_button"):
                    if not selected_tables:
                        st.warning("Please select at least one table to load.")
                    else:
                        with st.spinner(f"Loading data from {len(selected_tables)} table(s)... Please wait."):
                            loaded_dfs, successfully_loaded_names, messages = load_database_tables(
                                st.session_state.db_connection, selected_tables, max_rows_per_table
                            )
                            for msg in messages:
                                st.info(msg)
                            if loaded_dfs:
                                loaded_data_dict = {}
                                for i, df in enumerate(loaded_dfs):
                                    if i < len(successfully_loaded_names):
                                        loaded_data_dict[successfully_loaded_names[i]] = df
                                    else:
                                        loaded_data_dict[f"table_{i+1}"] = df # Fallback name
                                st.session_state.loaded_data = loaded_data_dict
                                st.session_state.loaded_sheet_names = successfully_loaded_names # Reusing this for table names
                                st.session_state.file_name = "PostgreSQL"
                                st.session_state.is_multisheet = len(loaded_dfs) > 1
                                initialize_analyzer()
                                if st.session_state.analyzer:
                                    st.success(f"Great! I've loaded table(s): **{', '.join(successfully_loaded_names)}** from PostgreSQL. You can now start chatting!")
                                    st.session_state.chat_history = []
                                    clear_analyzer_memory()
                                else:
                                    st.error("Failed to initialize the AI analyzer. Please check Google Cloud setup.")
                    #else:
                     #   st.error("Could not load data from the selected tables. Please check your selections and the database connection.")

        # Add additional options
        if 'loaded_sheet_names' in st.session_state and st.session_state.loaded_sheet_names:
            st.header("Options")
            if st.button("Clear Chat History"):
                clear_chat_history()
            if st.session_state.chat_history and st.button("Export Chat"):
                chat_text = "\n\n".join([f"**{role.upper()}**: {message}" for role, message in st.session_state.chat_history])
                st.download_button(
                    label="Download Chat",
                    data=chat_text,
                    file_name="data_analyzer_chat.md",
                    mime="text/markdown"
                )

        st.header("About")
        st.info("""
        This app analyzes Excel or PostgreSQL data using natural language queries.
        Powered by LangChain and Vertex AI.
        """)

    # Main area with chat interface
    if st.session_state.loaded_data is not None:
        st.header("Chat with Your Data")

        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        # Input area for new queries
        if prompt := st.chat_input("Ask a question about your data"):
            handle_query_submit(prompt)
    else:
        st.info("👈 Please upload an Excel file or connect to a PostgreSQL database from the sidebar to get started!")

if __name__ == "__main__":
    main()