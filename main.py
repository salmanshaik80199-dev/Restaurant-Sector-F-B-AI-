"""
Restaurant AI Assistant
A complete application for analyzing restaurant sales data using LangChain and Streamlit.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
except ImportError:
    try:
        from langchain.agents import create_pandas_dataframe_agent
    except ImportError:
        raise ImportError(
            "Please install langchain-experimental: pip install langchain-experimental\n"
            "The pandas DataFrame agent has been moved to langchain_experimental."
        )
from langchain_openai import ChatOpenAI

# Optional imports for different LLM providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_community.llms import Ollama  # For local LLM support
except ImportError:
    try:
        from langchain.llms import Ollama
    except ImportError:
        Ollama = None

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Restaurant AI Assistant",
    page_icon="🍽️",
    layout="wide"
)

# Initialize session state
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'cleaning_summary' not in st.session_state:
    st.session_state.cleaning_summary = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = None


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame to Arrow-compatible types for Streamlit display."""
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Handle dtype objects (like numpy.dtypes.Int64DType)
        if hasattr(df_copy[col].dtype, 'name') and 'Int' in str(df_copy[col].dtype):
            # Convert nullable integer types to float to handle NaN
            df_copy[col] = df_copy[col].astype('float64')
        # Convert nullable integer types to regular int or float
        elif pd.api.types.is_integer_dtype(df_copy[col]) and df_copy[col].isnull().any():
            # Convert nullable int to float to handle NaN
            df_copy[col] = df_copy[col].astype('float64')
        elif pd.api.types.is_integer_dtype(df_copy[col]):
            # Convert to regular int64
            df_copy[col] = df_copy[col].astype('int64')
        
        # Convert object columns (including dtype objects) to string
        if df_copy[col].dtype == 'object':
            try:
                # Convert to string, handling NaN and special objects
                df_copy[col] = df_copy[col].astype(str)
                df_copy[col] = df_copy[col].replace(['nan', 'None', '<NA>'], pd.NA)
            except Exception:
                # If conversion fails, try to convert each element
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if pd.notna(x) else pd.NA)
    
    return df_copy


class DataCleaner:
    """Handles all data cleaning operations for restaurant sales data."""
    
    def __init__(self):
        self.summary = {
            'original_rows': 0,
            'duplicates_removed': 0,
            'missing_values_handled': {},
            'data_type_fixes': [],
            'date_format_fixes': 0,
            'ingredient_normalizations': 0,
            'final_rows': 0,
            'cleaning_steps': []
        }
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file into pandas DataFrame."""
        try:
            df = pd.read_csv(file_path)
            self.summary['original_rows'] = len(df)
            self.summary['cleaning_steps'].append(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive data cleaning."""
        df_cleaned = df.copy()
        
        # Step 1: Normalize column names (lowercase, replace spaces with underscores)
        df_cleaned.columns = df_cleaned.columns.str.lower().str.strip().str.replace(' ', '_')
        self.summary['cleaning_steps'].append("Normalized column names to lowercase with underscores")
        
        # Step 2: Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        self.summary['duplicates_removed'] = duplicates_removed
        if duplicates_removed > 0:
            self.summary['cleaning_steps'].append(f"Removed {duplicates_removed} duplicate rows")
        
        # Step 3: Trim whitespace from string columns
        string_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in string_cols:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        self.summary['cleaning_steps'].append("Trimmed whitespace from all string columns")
        
        # Step 4: Handle missing values
        missing_before = df_cleaned.isnull().sum().to_dict()
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    self.summary['missing_values_handled'][col] = f"Filled with median: {median_val}"
                else:
                    # Fill string columns with 'Unknown' or mode
                    mode_val = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                    df_cleaned[col].fillna(mode_val, inplace=True)
                    self.summary['missing_values_handled'][col] = f"Filled with mode: {mode_val}"
        
        if self.summary['missing_values_handled']:
            self.summary['cleaning_steps'].append(f"Handled missing values in {len(self.summary['missing_values_handled'])} columns")
        
        # Step 5: Fix data types
        # Try to convert date columns
        date_columns = [col for col in df_cleaned.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                if df_cleaned[col].notna().sum() > 0:
                    self.summary['data_type_fixes'].append(f"Converted {col} to datetime")
                    self.summary['date_format_fixes'] += 1
            except:
                pass
        
        # Try to convert numeric columns
        numeric_columns = [col for col in df_cleaned.columns if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount', 'quantity', 'total', 'revenue', 'sales'])]
        for col in numeric_columns:
            if df_cleaned[col].dtype == 'object':
                try:
                    # Remove currency symbols and commas
                    df_cleaned[col] = df_cleaned[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('€', '').str.replace('£', '')
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    if df_cleaned[col].notna().sum() > 0:
                        self.summary['data_type_fixes'].append(f"Converted {col} to numeric")
                except:
                    pass
        
        # Step 6: Normalize ingredient fields
        ingredient_columns = [col for col in df_cleaned.columns if 'ingredient' in col.lower() or 'item' in col.lower() or 'product' in col.lower()]
        for col in ingredient_columns:
            if df_cleaned[col].dtype == 'object':
                # Normalize: lowercase, remove extra spaces
                df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
                df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
                self.summary['ingredient_normalizations'] += 1
        
        if self.summary['ingredient_normalizations'] > 0:
            self.summary['cleaning_steps'].append(f"Normalized {self.summary['ingredient_normalizations']} ingredient/item columns")
        
        # Step 7: Standardize date format (if date columns exist)
        for col in date_columns:
            if pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                # Ensure consistent datetime format
                df_cleaned[col] = pd.to_datetime(df_cleaned[col])
        
        self.summary['final_rows'] = len(df_cleaned)
        self.summary['cleaning_steps'].append(f"Final dataset has {len(df_cleaned)} rows and {len(df_cleaned.columns)} columns")
        
        return df_cleaned
    
    def get_summary(self) -> Dict:
        """Return cleaning summary."""
        return self.summary
    
    def save_cleaned_csv(self, df: pd.DataFrame, file_path: str = "cleaned_sales.csv") -> str:
        """Save cleaned DataFrame to CSV."""
        try:
            df.to_csv(file_path, index=False)
            return file_path
        except Exception as e:
            raise Exception(f"Error saving cleaned CSV: {str(e)}")


class LangChainAgentManager:
    """Manages LangChain agent creation and interaction."""
    
    @staticmethod
    def get_llm(llm_type: str, model_name: Optional[str] = None):
        """Get LLM instance based on user selection."""
        if llm_type == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            if model_name:
                return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
        
        elif llm_type == "Gemini":
            if ChatGoogleGenerativeAI is None:
                raise ValueError("langchain-google-genai is not installed. Install it with: pip install langchain-google-genai")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            if model_name:
                return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=api_key)
        
        elif llm_type == "Local":
            if Ollama is None:
                raise ValueError("Ollama is not installed. Install it with: pip install langchain-community ollama")
            if model_name:
                return Ollama(model=model_name, temperature=0)
            return Ollama(model="llama2", temperature=0)
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    @staticmethod
    def create_agent(df: pd.DataFrame, llm_type: str, model_name: Optional[str] = None):
        """
        Create a pandas DataFrame agent.
        
        Note: This agent requires allow_dangerous_code=True to execute pandas operations.
        The agent runs in a controlled environment with access only to the provided DataFrame.
        """
        try:
            llm = LangChainAgentManager.get_llm(llm_type, model_name)
            # langchain_experimental requires explicit opt-in for code execution
            try:
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    allow_dangerous_code=True,  # Required for pandas operations
                    return_intermediate_steps=True
                )
            except TypeError as e:
                # Fallback for different API versions
                try:
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True,
                        allow_dangerous_code=True,
                        return_intermediate_steps=True
                    )
                except TypeError:
                    # Minimal parameters (older versions)
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True
                    )
            return agent
        except Exception as e:
            raise Exception(f"Error creating agent: {str(e)}")


def main():
    """Main Streamlit application."""
    st.title("🍽️ Restaurant AI Assistant")
    st.markdown("Upload your restaurant sales data and get AI-powered insights!")
    
    # Sidebar for LLM selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        llm_type = st.selectbox(
            "Select LLM Provider",
            ["OpenAI", "Gemini", "Local"],
            help="Choose your preferred language model provider"
        )
        
        if llm_type == "OpenAI":
            model_name = st.text_input("Model Name (optional)", value="gpt-3.5-turbo", help="e.g., gpt-3.5-turbo, gpt-4")
        elif llm_type == "Gemini":
            model_name = st.text_input("Model Name (optional)", value="gemini-pro", help="e.g., gemini-pro, gemini-1.5-pro")
        else:  # Local
            model_name = st.text_input("Model Name (optional)", value="llama2", help="e.g., llama2, mistral")
        
        st.markdown("---")
        st.markdown("### 📋 Requirements")
        st.markdown("""
        - **OpenAI**: Set `OPENAI_API_KEY` in `.env` file
        - **Gemini**: Set `GOOGLE_API_KEY` in `.env` file
        - **Local**: Ensure Ollama is running locally
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Clean", "💬 Chat Assistant", "📊 Data Preview"])
    
    # Tab 1: Upload and Clean
    with tab1:
        st.header("Upload and Clean Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your restaurant sales data CSV file"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and display raw data info
                cleaner = DataCleaner()
                raw_df = cleaner.load_csv("temp_upload.csv")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", len(raw_df))
                with col2:
                    st.metric("Total Columns", len(raw_df.columns))
                
                st.subheader("Raw Data Preview")
                st.dataframe(make_arrow_compatible(raw_df.head(10)), width='stretch')
                
                # Clean data button
                if st.button("🧹 Clean Data", type="primary"):
                    with st.spinner("Cleaning data... This may take a moment."):
                        cleaned_df = cleaner.clean_data(raw_df)
                        summary = cleaner.get_summary()
                        
                        # Save cleaned data
                        output_path = cleaner.save_cleaned_csv(cleaned_df)
                        
                        # Store in session state
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_summary = summary
                        
                        st.success(f"✅ Data cleaned successfully! Saved as `{output_path}`")
                        
                        # Display cleaning summary
                        st.subheader("📋 Cleaning Summary")
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.metric("Original Rows", summary['original_rows'])
                            st.metric("Duplicates Removed", summary['duplicates_removed'])
                            st.metric("Final Rows", summary['final_rows'])
                        
                        with summary_col2:
                            st.metric("Date Format Fixes", summary['date_format_fixes'])
                            st.metric("Ingredient Normalizations", summary['ingredient_normalizations'])
                            st.metric("Data Type Fixes", len(summary['data_type_fixes']))
                        
                        # Show cleaning steps
                        st.subheader("🔧 Cleaning Steps Performed")
                        for i, step in enumerate(summary['cleaning_steps'], 1):
                            st.write(f"{i}. {step}")
                        
                        # Show missing values handled
                        if summary['missing_values_handled']:
                            st.subheader("🔍 Missing Values Handled")
                            for col, action in summary['missing_values_handled'].items():
                                st.write(f"- **{col}**: {action}")
                        
                        # Initialize agent with cleaned data
                        try:
                            agent = LangChainAgentManager.create_agent(cleaned_df, llm_type, model_name)
                            st.session_state.agent = agent
                            st.success("🤖 AI Agent initialized successfully!")
                        except Exception as e:
                            st.error(f"Error initializing agent: {str(e)}")
                            st.info("Please check your API keys in the .env file")
                
                # Clean up temp file
                if os.path.exists("temp_upload.csv"):
                    os.remove("temp_upload.csv")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Tab 2: Chat Assistant
    with tab2:
        st.header("💬 Chat with AI Assistant")
        
        if st.session_state.cleaned_df is None:
            st.warning("⚠️ Please upload and clean your data first in the 'Upload & Clean' tab.")
            st.info("Example questions you can ask once data is loaded:")
            st.markdown("""
            - "What were the top-selling items last week?"
            - "How much ingredients should I order next week?"
            - "What caused the drop in sales on Tuesday?"
            - "Show me the total revenue by day"
            - "Which items have the highest profit margin?"
            """)
        else:
            # Display data info
            st.info(f"📊 Working with {len(st.session_state.cleaned_df)} rows of cleaned data")
            
            # Chat interface
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # User input
            user_query = st.chat_input("Ask a question about your restaurant data...")
            
            if user_query:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)
                
                # Get AI response
                if st.session_state.agent is None:
                    try:
                        agent = LangChainAgentManager.create_agent(
                            st.session_state.cleaned_df,
                            llm_type,
                            model_name
                        )
                        st.session_state.agent = agent
                    except Exception as e:
                        st.error(f"Error creating agent: {str(e)}")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Sorry, I couldn't initialize the AI agent. Error: {str(e)}"
                        })
                        return
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Use invoke() instead of deprecated run()
                            # When return_intermediate_steps=True, invoke returns a dict
                            # Try different input formats for compatibility
                            try:
                                result = st.session_state.agent.invoke({"input": user_query})
                            except (TypeError, KeyError):
                                # Some agents accept the query directly
                                result = st.session_state.agent.invoke(user_query)
                            
                            # Extract the output from the result
                            if isinstance(result, dict):
                                # Handle dict response (when return_intermediate_steps=True)
                                response = result.get("output", result.get("answer", str(result)))
                            else:
                                response = result
                            
                            st.write(response)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": str(response)
                            })
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Tab 3: Data Preview
    with tab3:
        st.header("📊 Cleaned Data Preview")
        
        if st.session_state.cleaned_df is None:
            st.warning("⚠️ No cleaned data available. Please upload and clean your data first.")
        else:
            df = st.session_state.cleaned_df
            
            # Data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns.astype(str),
                'Data Type': [str(dt) for dt in df.dtypes],  # Convert dtype objects to strings
                'Non-Null Count': df.count().astype('int64'),
                'Null Count': df.isnull().sum().astype('int64')
            })
            st.dataframe(make_arrow_compatible(col_info), width='stretch')
            
            # Data preview
            st.subheader("Data Preview")
            num_rows = st.slider("Number of rows to display", 5, min(100, len(df)), 10)
            st.dataframe(make_arrow_compatible(df.head(num_rows)), width='stretch')
            
            # Download cleaned data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="cleaned_sales.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

