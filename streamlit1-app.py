
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
import time # Import the time module

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
                temperature=0.7, # Keep temperature a bit higher for more creative responses
                max_output_tokens=2000
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            logger.info(f"Data analyzer initialized successfully with project ID: {project_id}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def analyze_data_stream(self, df, query):
        """
        Generate analysis of the dataframe based on the user query, streaming the response.
        
        Args:
            df (pandas.DataFrame): The dataframe to analyze
            query (str): The user's question
            
        Yields:
            str: Chunks of the analysis result
        """
        try:
            if df is None or df.empty:
                yield "It looks like there's no data loaded yet. Please upload an Excel file first so I can start analyzing!"
                return
            
            # Create a pandas agent for data analysis with code execution enabled
            # You can try to guide the agent's persona through the system message,
            # but for create_pandas_dataframe_agent, it's primarily through the 'input' query.
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                verbose=False, # Set to False for cleaner Streamlit output; logs go to console
                memory=self.memory,
                allow_dangerous_code=True  # Enable code execution for data analysis
            )
            
            # Stream the response
            # LangChain's agent.stream yields dictionaries, we're interested in 'output'
            # or 'final_answer' depending on the agent's internal steps.
            # For pandas agent, 'output' usually contains the final response.
            full_output_generated = False
            for chunk in agent.stream({"input": query}):
                if "output" in chunk:
                    yield chunk["output"]
                    full_output_generated = True
                # If you want to stream intermediate thoughts/tool usage, you'd check other keys
                # e.g., if "intermediate_steps" in chunk: yield f"\n_Thinking: {chunk['intermediate_steps']}_\n"
            
            if not full_output_generated:
                yield "I processed your request, but didn't receive a specific output. Could you try rephrasing your question?"
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            error_msg = f"Oops! I encountered an error during analysis: {str(e)}. Could you please check your query or the data?"
            logger.error(error_msg)
            yield error_msg

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
        # Reset file pointer for subsequent reads
        uploaded_file.seek(0) 
        if sheet_name:
            header_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=0)
        else:
            header_df = pd.read_excel(uploaded_file, nrows=0)
            
        column_count = len(header_df.columns)
        
        # Check if file is too large (rough estimation)
        if column_count > 200:
            st.warning(f"This file has **{column_count} columns**, which might cause performance issues. For optimal experience, consider using a dataset with fewer columns or a smaller number of rows.")
        
        # Reset file pointer before reading full data
        uploaded_file.seek(0)
        # Load the actual data with a row limit
        if sheet_name:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=max_rows)
        else:
            df = pd.read_excel(uploaded_file, nrows=max_rows)
            
        if len(df) == max_rows:
            st.warning(f"Just a heads-up: I've loaded the first **{max_rows} rows** of your data to prevent memory issues. If your analysis requires the full dataset, please consider a smaller file or a more powerful environment.")
            
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
                    pass # Keep as object if conversion fails or is not appropriate
                    
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}")
        st.error(f"Darn! I couldn't load your Excel file. Here's what went wrong: {str(e)}. Please make sure it's a valid .xlsx or .xls file and not corrupted.")
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

def handle_query_submit(query):
    """Handle submission of a query in the chat interface."""
    if not query:
        return

    if st.session_state.df is None:
        st.error("Hold on! Please **upload and load a dataset** first in the sidebar before asking questions.")
        return

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Use a placeholder for the AI's streaming response
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response_content = []

        # Stream the response from the analyzer word by word
        for chunk_from_analyzer in st.session_state.analyzer.analyze_data_stream(st.session_state.df, query):
            if chunk_from_analyzer: # Ensure chunk is not empty
                words = chunk_from_analyzer.split(" ") # Split by space
                for i, word in enumerate(words):
                    full_response_content.append(word)
                    if i < len(words) - 1: # Add space after word, but not the very last one
                        full_response_content.append(" ")
                    response_placeholder.markdown("".join(full_response_content) + "▌") # Add blinking cursor
                    time.sleep(0.05) # Adjust this value, e.g., 0.05 to 0.1 for word speed

                # Add a tiny delay after processing a full chunk from the LLM,
                # in case the chunk ends mid-sentence.
                time.sleep(0.02)

        final_response_text = "".join(full_response_content).strip() # Remove any potential trailing space
        response_placeholder.markdown(final_response_text) # Display final text without cursor

    # Add complete AI response to chat history
    st.session_state.chat_history.append({"role": "ai", "content": final_response_text})

    
def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []
    st.success("Chat history cleared! Ready for a new conversation.")

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
    st.title("📊 Chat with your Excel Data")
    st.markdown("""
    Upload an Excel file and let me help you **analyze your data** using natural language!
    Ask me questions, and I'll do my best to provide insights.
    """)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("1. Upload Excel File")
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            # Get list of sheet names
            try:
                # Reset file pointer for initial read
                uploaded_file.seek(0) 
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                
                st.header("2. Select Sheet & Load")
                selected_sheet = st.selectbox("Which sheet would you like to analyze?", sheet_names, index=0)
                
                # Add a row limit option
                max_rows = st.number_input("Maximum rows to load (for performance)", 
                                             min_value=100, 
                                             max_value=500000, # Increased max limit slightly
                                             value=100000,
                                             help="Limit the number of rows to prevent memory issues with very large files.")
                
                if st.button("Load Data into AI"):
                    with st.spinner("Loading and processing your data..."):
                        df = load_excel(uploaded_file, selected_sheet, max_rows)
                        
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.file_name = uploaded_file.name
                            st.session_state.sheet_name = selected_sheet
                            
                            # Initialize analyzer if not already done
                            if st.session_state.analyzer is None:
                                try:
                                    st.session_state.analyzer = DataAnalyzer()
                                except Exception as e:
                                    st.error(f"Failed to initialize the AI analyzer. Error: {e}. Please ensure your Google Cloud environment is properly configured (e.g., `gcloud auth application-default login`).")
                                    st.session_state.analyzer = None # Ensure analyzer is None if init fails
                                    return # Stop further execution
                            
                            st.success(f"Great! I've loaded **{uploaded_file.name}** (Sheet: **{selected_sheet}**) for analysis. You can now start chatting!")
                            
                            # Clear chat history when loading new data
                            st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Oh no, I couldn't read your Excel file or its sheets. Error: {str(e)}. Is it a valid Excel format?")
        
        # Add additional options
        if st.session_state.df is not None:
            st.markdown("---")
            st.header("Chat Options")
            
            if st.button("Start New Chat", help="Clear the current conversation history"):
                clear_chat_history()
            
            # Add option to export chat history
            if st.session_state.chat_history and st.button("Download Chat Log", help="Save the conversation as a Markdown file"):
                chat_text_content = []
                for entry in st.session_state.chat_history:
                    role_display = "You" if entry["role"] == "user" else "AI"
                    chat_text_content.append(f"**{role_display}**: {entry['content']}")
                
                st.download_button(
                    label="Download Chat",
                    data="\n\n".join(chat_text_content),
                    file_name="excel_analyzer_chat_log.md",
                    mime="text/markdown"
                )
        
        st.markdown("---")
        st.header("About This App")
        st.info("""
        This interactive tool helps you analyze Excel data effortlessly using the power of **Google's Vertex AI (Gemini 1.5 Pro)** and **LangChain**.
        
        Simply upload your data, and ask questions in plain English to get insights, summaries, and more!
        """)
    
    # Main area with tabs for data preview and chat interface
    if st.session_state.df is not None:
        # Create tabs
        chat_tab, data_tab = st.tabs(["💬 Chat with Data", "📊 Data Preview & Summary"])
        
        # Chat interface tab
        with chat_tab:
            st.header(f"Chatting about: {st.session_state.file_name} (Sheet: {st.session_state.sheet_name})")
            st.markdown("Feel free to ask me anything about your loaded data. For example: 'What are the average sales?', 'Show me the distribution of customer ages.', or 'Identify any outliers in the 'Price' column.'")
            
            # Display chat history using st.chat_message
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input area for new queries
            # Use st.chat_input for a more native chat experience
            query_input = st.chat_input(
                "Type your question about the data here...",
                key="query_input"
            )
            
            if query_input: # st.chat_input handles submission directly
                # Ensure the analyzer is initialized before handling queries
                if st.session_state.analyzer is None:
                    try:
                        st.session_state.analyzer = DataAnalyzer()
                    except Exception as e:
                        st.error(f"Failed to initialize the AI analyzer. Error: {e}. Please check your Google Cloud setup.")
                        st.session_state.analyzer = None 
                        return 
                
                handle_query_submit(query_input)
                # No st.rerun() needed for st.chat_input; it handles state itself after submission.
                # However, for a more immediate chat update, a rerun might still be desired
                # if other parts of the app depend on the chat history being immediately refreshed.
                # For basic chat, it's often not strictly necessary. Let's remove it for now.
                # st.rerun() # Removed as chat_input often manages its own state
        
        # Data preview and summary tab
        with data_tab:
            st.header("Your Data: Preview & Summary")
            st.markdown(f"**Currently analyzing:** `{st.session_state.file_name}` (Sheet: `{st.session_state.sheet_name}`)")
            
            # Display the dataframe with pagination
            st.subheader("Data Table Preview (First 10 Rows)")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
            # Display additional data information
            st.subheader("Comprehensive Data Summary")
            display_data_info(st.session_state.df)
    else:
        # Show welcome message when no data is loaded
        st.info("👋 Welcome! To get started, please **upload an Excel file** from the sidebar on the left.")
        
        # Show example usage
        st.markdown("---")
        with st.expander("🤔 How to use this app"):
            st.markdown("""
            ### **Your Interactive Excel Analyzer**
            
            This app is designed to make data analysis super easy. Here's how to use it:
            
            1.  **Upload Your Excel File:** Find the "Upload Excel File" section in the sidebar on the left. Click "Browse files" and select your `.xlsx` or `.xls` document.
            2.  **Choose Your Sheet:** If your Excel file has multiple sheets, a dropdown will appear. Select the one you want to analyze.
            3.  **Load the Data:** Click the "Load Data into AI" button. I'll process your file, and you'll see a success message once it's ready.
            4.  **Start Chatting!** Once loaded, the "Chat with Data" tab will become active. Type your questions about the data in the chat input box at the bottom.
            
            ---
            
            ### **What can you ask?**
            
            I'm powered by advanced AI, so you can ask a wide range of questions in natural language. Here are some ideas:
            
            * "**Summarize the key characteristics** of this dataset."
            * "What is the **average 'Sales' amount**?"
            * "Show me the **distribution of 'Customer_Age'**."
            * "Are there any **outliers in the 'Revenue' column**? If so, what are they?"
            * "**Compare the 'Profit' between different 'Product_Categories'**."
            * "Which **'Region' has the highest total 'Orders'**?"
            * "Can you **identify any trends** in the 'Date' and 'Value' columns?"
            * "What are the **unique values** in the 'Status' column?"
            * "**Clean the 'Customer_Name' column** by removing extra spaces." (You can even ask for data cleaning suggestions!)
            
            Feel free to be creative with your questions!
            """)
    
    # Footer
    st.markdown("---")
    st.caption("✨ Powered by Streamlit, LangChain, and Google Vertex AI (Gemini 1.5 Pro)")

if __name__ == "__main__":
    main()