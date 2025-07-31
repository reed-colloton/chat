import os
import json
import datetime
import pickle
import tempfile

import streamlit as st
from openai import AzureOpenAI

import prompts

# Configure page
st.set_page_config(
    page_title="PairD 2",
    page_icon="🤖",
    initial_sidebar_state="auto"
)

# Header
st.header("PairD 2", divider="gray", anchor=False)

# Configuration
azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = "https://reedc-mdrjnjlt-swedencentral.cognitiveservices.azure.com/"

# Available models
AVAILABLE_MODELS = {
    "GPT 4o": "gpt-4o",
    "GPT 4.1": "gpt-4.1",
    "GPT 3.5": "gpt-35-turbo",
    "o4 mini": "o4-mini",
}

# Initialize model selection in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT 4o"

# Get current model
model = AVAILABLE_MODELS[st.session_state.selected_model]
api_version = "2024-12-01-preview"

# Session persistence functions
def get_session_file_path():
    """Get the path for the session persistence file"""
    temp_dir = tempfile.gettempdir()

    # Use a more stable session ID based on Streamlit's session info
    if 'session_id' not in st.session_state:
        # Create a persistent session ID that survives page refreshes
        session_hash = hash(str(id(st.session_state))) % 1000000
        st.session_state.session_id = f"paird_chat_{session_hash}"

    session_file = os.path.join(temp_dir, f"{st.session_state.session_id}.pkl")
    return session_file

def save_session_data():
    """Save session data to temporary file"""
    try:
        if not st.session_state.get('messages'):
            return True  # Nothing to save

        session_file = get_session_file_path()
        session_data = {
            'messages': st.session_state.messages,
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': st.session_state.session_id
        }

        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)

        # Debug info
        st.session_state.last_save_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.save_file_path = session_file
        return True
    except Exception as e:
        st.session_state.save_error = f"Save failed: {str(e)}"
        return False

def load_session_data():
    """Load session data from temporary file"""
    try:
        session_file = get_session_file_path()

        if os.path.exists(session_file):
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)

            # Always load the messages if the file exists
            if 'messages' in session_data and session_data['messages']:
                return session_data['messages'], len(session_data['messages'])

        return [], 0
    except Exception as e:
        st.session_state.load_error = f"Load failed: {str(e)}"
        return [], 0

def cleanup_session_data():
    """Clean up session data file"""
    try:
        session_file = get_session_file_path()
        if os.path.exists(session_file):
            os.remove(session_file)
        return True
    except Exception:
        return False

# Helper functions for chat history
def save_chat_history():
    """Save current chat history to a JSON file"""
    if st.session_state.messages:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_data = {
            "timestamp": timestamp,
            "messages": st.session_state.messages,
            "model": st.session_state.selected_model,
            "total_messages": len(st.session_state.messages)
        }

        # Create JSON string
        json_string = json.dumps(chat_data, indent=2, ensure_ascii=False)

        # Create download button in session state for later use
        st.session_state.download_data = json_string
        st.session_state.download_filename = f"chat_history_{timestamp}.json"
        return True
    return False

def load_chat_history(uploaded_file):
    """Load chat history from uploaded JSON file"""
    try:
        # Read the uploaded file
        content = uploaded_file.read().decode('utf-8')
        chat_data = json.loads(content)

        # Validate the data structure
        if "messages" in chat_data and isinstance(chat_data["messages"], list):
            st.session_state.messages = chat_data["messages"]
            st.session_state.conversation_loaded = True
            return True, f"✅ Loaded {len(chat_data['messages'])} messages from {chat_data.get('timestamp', 'Unknown date')}"
        else:
            return False, "❌ Invalid chat history file format"
    except json.JSONDecodeError:
        return False, "❌ Invalid JSON file"
    except Exception as e:
        return False, f"❌ Error loading file: {str(e)}"

def export_chat_as_text():
    """Export chat history as readable text"""
    if st.session_state.messages:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        text_content = f"Chat History Export\n"
        text_content += f"Exported on: {timestamp}\n"
        text_content += f"Model: {st.session_state.selected_model}\n"
        text_content += f"Total Messages: {len(st.session_state.messages)}\n"
        text_content += "=" * 50 + "\n\n"

        for i, message in enumerate(st.session_state.messages, 1):
            role = "You" if message["role"] == "user" else "AI Assistant"
            text_content += f"[{i}] {role}:\n{message['content']}\n\n"

        return text_content
    return None

# Sidebar configuration
with st.sidebar:
    # Model selection
    selected_model = st.selectbox(
        "",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
        key="model_selector"
    )

    # Update session state if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()

    # Add some spacing
    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        # Clear download data when clearing chat
        if hasattr(st.session_state, 'download_data'):
            del st.session_state.download_data
        if hasattr(st.session_state, 'export_text'):
            del st.session_state.export_text
        # Clear session persistence file
        cleanup_session_data()
        st.rerun()



# Check for API key
if not azure_api_key:
    st.error("🔑 **Azure API Key not found!** Please set the AZURE_API_KEY environment variable.")
    st.stop()

# Create an OpenAI client with error handling
try:
    client = AzureOpenAI(api_key=azure_api_key, azure_endpoint=azure_endpoint, api_version=api_version)
except Exception as e:
    st.error(f"❌ **Failed to initialize OpenAI client:** {str(e)}")
    st.stop()

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session persistence
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True

    # Try to load previous session data
    loaded_messages, message_count = load_session_data()

    if loaded_messages:
        st.session_state.messages = loaded_messages
        st.session_state.session_restored = True
        st.session_state.restored_message_count = message_count
    else:
        st.session_state.messages = []

# Main chat container
chat_container = st.container()

# Display welcome message if no messages exist
if not st.session_state.messages:
    with chat_container:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #f8fafc; border-radius: 15px; margin: 1rem 0;">
            <p style="color: #6b7280; font-size: 1.1rem;">
                I'm here to help! This conversation is secure and private.
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Show conversation info if available
    if hasattr(st.session_state, 'conversation_loaded') and st.session_state.conversation_loaded:
        st.info("📁 **Conversation loaded from file** - Continue where you left off!")
        # Reset the flag so it doesn't show again
        st.session_state.conversation_loaded = False


# Display the existing chat messages with improved styling
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar=":material/account_circle:"):
                st.markdown(f"{message['content']}")
        else:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Ask anything", key="chat_input"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Save session data after user input
    save_session_data()

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"{prompt}")

    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                # Generate a response using the OpenAI API.
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompts.SYSTEM_PROMPT.format(
                                current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                                model=model

                            )
                        },
                        *[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    ],
                    stream=True,
                )

                # Stream the response to the chat using `st.write_stream`, then store it in
                # session state.
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Save session data after each interaction
                save_session_data()

            except Exception as e:
                error_message = f"❌ **Error generating response:** {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

                # Save session data even after errors
                save_session_data()
