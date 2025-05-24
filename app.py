import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from google.auth import default
from dotenv import load_dotenv
from sub_agent import generate_sub_questions,generate_consolidated_answer
import logging
import os
import gc
import time
from tenacity import retry, stop_after_attempt, wait_exponential

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

    @retry(
        stop=stop_after_attempt(5),  # Stop after 5 attempts
        wait=wait_exponential(multiplier=1, min=4, max=60),  # Wait between 4 and 60 seconds, exponentially increasing
        reraise=True
    )
    def _run_agent_with_retry(self, agent, question):
        """Execute agent run with retry logic."""
        try:
            return agent.run(question)
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e):
                logger.warning(f"Rate limit hit, retrying: {str(e)}")
                raise  # Retry on rate limit
            raise  # Don't retry on other errors

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
            
            # Generate sub-questions with retry
            try:
                questions_dict = generate_sub_questions(query, list(df.columns), self.llm)
            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    logger.warning("Rate limit hit during question generation, waiting 10 seconds...")
                    time.sleep(10)
                    questions_dict = generate_sub_questions(query, list(df.columns), self.llm)
                else:
                    raise
            
            # Create a pandas agent for data analysis with code execution enabled
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=True,
                memory=self.memory,
                allow_dangerous_code=True
            )
            
            # Generate responses for all questions with retry logic
            responses = []
            for question in questions_dict['questions']:
                try:
                    response = self._run_agent_with_retry(agent, question)
                    responses.append(f"For question: {question}\nAnswer: {response}\n")
                    logger.info(f"Analysis completed for question: {question}")
                except Exception as e:
                    responses.append(f"For question: {question}\nError: Failed to get response after retries: {str(e)}\n")
                    logger.error(f"Failed to get response for question after retries: {question}")
                
                # Add a small delay between questions to avoid rate limiting
                time.sleep(2)
            
            # Combine all responses
            final_response = generate_consolidated_answer("\n".join(responses),self.llm)
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while analyzing your data: {str(e)}"

def load_excel(uploaded_file, sheet_name=None, max_rows=100000):
    """
    Load Excel file and return DataFrame with size limit.
    
    Args:
        uploaded_file: The uploaded file object
        sheet_name (str, optional): Name of the sheet to load
        max_rows (int): Maximum number of rows to load to prevent memory issues
        
    Returns:
        pandas.DataFrame: Loaded dataframe or None if loading fails
    """
    try:
        # First load just the header to check column count
        if sheet_name:
            header_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=0)
        else:
            header_df = pd.read_excel(uploaded_file, nrows=0)
            
        column_count = len(header_df.columns)
        
        # Check if file is too large (rough estimation)
        if column_count > 200:
            st.warning(f"This file has {column_count} columns which might cause performance issues. Consider using a smaller dataset.")
        
        # Load the actual data with a row limit
        if sheet_name:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=max_rows)
        else:
            df = pd.read_excel(uploaded_file, nrows=max_rows)
            
        if len(df) == max_rows:
            st.warning(f"Only loaded first {max_rows} rows to prevent memory issues.")
            
        logger.info(f"Successfully loaded Excel file with shape: {df.shape}")
        
        # Convert problematic data types that might cause issues
        for col in df.columns:
            # Handle object columns with mixed types
            if df[col].dtype == 'object':
                # Try to convert to numeric if appropriate
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    # If successful and not too many NaNs introduced, use it
                    if numeric_col.isna().sum() < 0.5 * len(df):
                        df[col] = numeric_col
                except:
                    pass
                    
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}")
        return None
    finally:
        # Force garbage collection to free memory
        gc.collect()

def display_data_info(df):
    """
    Display information about the dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
    """
    if df is None or df.empty:
        st.warning("No data available to display")
        return
        
    st.subheader("Data Summary")
    
    # Basic dataframe stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isna().sum().sum())
    
    # Column information
    with st.expander("Column Details"):
        column_info = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(dt) for dt in df.dtypes.values],
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(column_info, use_container_width=True)
    
    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        with st.expander("Numeric Column Statistics"):
            try:
                stats_df = df[numeric_cols].describe()
                st.dataframe(stats_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying statistics: {str(e)}")

def initialize_session_state():
    """Initialize all session state variables needed for the app."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'df' not in st.session_state:
        st.session_state.df = None
        
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
        
    if 'sheet_name' not in st.session_state:
        st.session_state.sheet_name = None
    
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
        
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    if 'submit_query' not in st.session_state:
        st.session_state.submit_query = False

def submit_on_enter():
    st.session_state.submit_query = True

def handle_query_submit(query):
    """Handle submission of a query in the chat interface."""
    if not query:
        return
        
    if st.session_state.df is None:
        st.error("Please upload and load a dataset first.")
        return
        
    # Add user message to chat history
    st.session_state.chat_history.append(("user", query))
    
    # Show processing indicator
    with st.spinner("Analyzing your data..."):
        # Process the query
        response = st.session_state.analyzer.analyze_data(st.session_state.df, query)
    
    # Add AI response to chat history
    st.session_state.chat_history.append(("ai", response))

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []

def main():
    """Main function to run the Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="Excel Analyzer Chatbot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # App title and description
    st.title("📊 Excel Analyzer Chatbot")
    st.markdown("""
    Upload an Excel file and ask questions about your data in natural language.
    The AI will analyze your data and provide insights based on your queries.
    """)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Excel File")
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            # Get list of sheet names
            try:
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                
                selected_sheet = st.selectbox("Select a sheet", sheet_names, index=0)
                
                # Add a row limit option
                max_rows = st.number_input("Maximum rows to load", 
                                          min_value=100, 
                                          max_value=1000000, 
                                          value=100000,
                                          help="Limit the number of rows to prevent memory issues")
                
                if st.button("Load Data"):
                    with st.spinner("Loading data..."):
                        # Load data
                        df = load_excel(uploaded_file, selected_sheet, max_rows)
                        
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.file_name = uploaded_file.name
                            st.session_state.sheet_name = selected_sheet
                            
                            # Initialize analyzer if not already done
                            if st.session_state.analyzer is None:
                                st.session_state.analyzer = DataAnalyzer()
                            
                            st.success(f"Successfully loaded {uploaded_file.name}, sheet: {selected_sheet}")
                            
                            # Clear chat history when loading new data
                            st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
        
        # Add additional options
        if st.session_state.df is not None:
            st.header("Options")
            
            if st.button("Clear Chat History"):
                clear_chat_history()
            
            # Add option to export chat history
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
        This app allows you to analyze Excel data using natural language queries.
        Upload an Excel file, select a sheet, and start asking questions about your data.
        
        Powered by LangChain and Vertex AI.
        """)
    
    # Main area with tabs for data preview and chat interface
    if st.session_state.df is not None:
        # Create tabs
        chat_tab, data_tab = st.tabs(["💬 Chat with Data", "📊 Data Preview & Summary"])
        
        # Chat interface tab
        with chat_tab:
            st.header("Chat with Your Data")
            
            # Create a container for chat messages with scrolling
            chat_container = st.container()
            
            # Display chat history
            with chat_container:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.markdown(f"**You**: {message}")
                    else:
                        st.markdown(f"**AI**: {message}")
                    
                    # Add a separator between messages
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
            
            # Input area for new queries
            query_form = st.form(key="query_form")
            query_input = st.text_area(
                                    "Ask a question about your data:",
                                    key="query_input",
                                    height=100,
                                    placeholder="e.g., Analyze the data and summarize key insights",
                                    on_change=submit_on_enter
                                )
            
            col1, col2 = query_form.columns([1, 5])
            with col1:
                submit_button = query_form.form_submit_button("Send", use_container_width=True)
            
            if (submit_button or st.session_state.submit_query) and query_input:
                handle_query_submit(query_input)
                # Reset the submit state
                st.session_state.submit_query = False
                # Clear the form by rerunning the app
                st.rerun()
        
        # Data preview and summary tab
        with data_tab:
            st.header("Data Preview & Summary")
            st.write(f"File: {st.session_state.file_name} | Sheet: {st.session_state.sheet_name}")
            
            # Display the dataframe with pagination
            st.subheader("Data Preview")
            st.write("First 10 rows:")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
            # Display additional data information
            st.subheader("Data Summary")
            display_data_info(st.session_state.df)
    else:
        # Show welcome message when no data is loaded
        st.info("👈 Please upload an Excel file from the sidebar to get started!")
        
        # Show example usage
        with st.expander("How to use this app"):
            st.markdown("""
            ### How to use the Excel Analyzer Chatbot
            
            1. **Upload your Excel file** using the file uploader in the sidebar
            2. **Select a sheet** if your Excel file has multiple sheets
            3. **Click "Load Data"** to load the selected sheet
            4. **Ask questions** about your data in natural language
            5. The AI will analyze your data and provide insights
            
            ### Example questions you can ask:
            
            - "Summarize the key insights from this data"
            - "What are the trends in this dataset?"
            - "Find the top 5 values in column X and explain why they stand out"
            - "Compare the values between different categories"
            - "Identify outliers in the data and explain their significance"
            """)
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit, LangChain, and Vertex AI")

if __name__ == "__main__":
    main()