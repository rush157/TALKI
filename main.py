import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import langdetect
from datetime import datetime
import re
import sentencepiece  # Required for M2M100 model

# Page configuration
st.set_page_config(
    page_title="TALKI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2E86AB;
    }
    .user-message {
        background-color: black;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: black;
        margin-right: 2rem;
    }
    .language-badge {
        background-color: #2E86AB;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Language mapping - Updated for M2M100 compatibility
LANGUAGE_CODES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Dutch': 'nl',
    'Swedish': 'sv',
    'Norwegian': 'no',
    'Danish': 'da',
    'Finnish': 'fi',
    'Polish': 'pl',
    'Czech': 'cs',
    'Hungarian': 'hu'
}

# M2M100 specific language codes
M2M100_LANG_CODES = {
    'en': 'en',
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'it': 'it',
    'pt': 'pt',
    'ru': 'ru',
    'zh': 'zh',
    'ja': 'ja',
    'ko': 'ko',
    'ar': 'ar',
    'hi': 'hi',
    'nl': 'nl',
    'sv': 'sv',
    'no': 'no',
    'da': 'da',
    'fi': 'fi',
    'pl': 'pl',
    'cs': 'cs',
    'hu': 'hu'
}

@st.cache_resource
def load_translation_model():
    """Load the translation model with caching"""
    try:
        # Using Facebook's M2M100 model for multilingual translation
        model_name = "facebook/m2m100_418M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def detect_language(text):
    """Detect the language of input text"""
    try:
        detected = langdetect.detect(text)
        # Map langdetect codes to language names
        lang_map = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh-cn': 'Chinese',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 
            'hi': 'Hindi', 'nl': 'Dutch', 'sv': 'Swedish', 'no': 'Norwegian', 
            'da': 'Danish', 'fi': 'Finnish', 'pl': 'Polish', 'cs': 'Czech', 
            'hu': 'Hungarian'
        }
        return lang_map.get(detected, detected.upper())
    except Exception as e:
        st.warning(f"Language detection failed: {str(e)}")
        return "English"  # Default to English if detection fails

def translate_text(text, source_lang, target_lang, tokenizer, model):
    """Translate text using the loaded model"""
    try:
        # Set source language
        tokenizer.src_lang = source_lang
        
        # Encode the input text
        encoded = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        # Decode the translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return f"Sorry, I couldn't translate that text. Error: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'model' not in st.session_state:
        st.session_state.model = None

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌍 TALKI - Translation Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading status
        if st.session_state.tokenizer is None or st.session_state.model is None:
            with st.spinner("Loading translation model... This may take a few minutes on first run."):
                tokenizer, model = load_translation_model()
                if tokenizer and model:
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model. Please check your internet connection and try refreshing.")
                    st.stop()
        else:
            st.success("✅ Model ready!")
        
        # Language selection
        st.subheader("🗣️ Language Settings")
        
        auto_detect = st.checkbox("Auto-detect source language", value=True)
        
        if not auto_detect:
            source_lang = st.selectbox(
                "Source Language",
                options=list(LANGUAGE_CODES.keys()),
                index=0
            )
        
        target_lang = st.selectbox(
            "Target Language",
            options=list(LANGUAGE_CODES.keys()),
            index=1  # Default to Spanish
        )
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Statistics
        if st.session_state.messages:
            st.subheader("📊 Chat Statistics")
            total_messages = len(st.session_state.messages)
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.metric("Total Messages", total_messages)
            st.metric("Translations Made", user_messages)
        
        # Info section
        st.subheader("ℹ️ About TALKI")
        st.info("""
        **TALKI** is your multilingual translation assistant!
        
        **Features:**
        - 🔍 Auto language detection
        - 🌐 20+ supported languages
        - 🤖 Powered by Facebook's M2M100 model
        - ⚡ Real-time translation
        - 💬 Interactive chat interface
        
        **Usage:**
        Simply type any text and it will be translated to your selected target language.
        """)
    
    # Main chat interface
    st.subheader("💬 Chat with TALKI")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>
                <span class="language-badge">{message.get('detected_lang', 'Unknown')}</span>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>TALKI:</strong><br>
                <span class="language-badge">{message.get('target_lang', 'Unknown')}</span>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message to translate... 🌍")
    
    if user_input and user_input.strip():
        # Detect or use selected source language
        if auto_detect:
            detected_lang = detect_language(user_input)
            source_lang_code = LANGUAGE_CODES.get(detected_lang, 'en')
        else:
            detected_lang = source_lang
            source_lang_code = LANGUAGE_CODES[source_lang]
        
        target_lang_code = LANGUAGE_CODES[target_lang]
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "detected_lang": detected_lang
        })
        
        # Perform translation
        with st.spinner("🔄 Translating..."):
            if source_lang_code == target_lang_code:
                translation = f"✨ The text is already in {target_lang}. No translation needed!\n\n📝 Original: {user_input}"
            else:
                translation = translate_text(
                    user_input,
                    source_lang_code,
                    target_lang_code,
                    st.session_state.tokenizer,
                    st.session_state.model
                )
        
        # Add bot response to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": translation,
            "target_lang": target_lang
        })
        
        # Rerun to update the chat display
        st.rerun()
    
    # Empty state message
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>👋 Welcome to TALKI!</h3>
            <p>Start by typing any text in the chat box below. I'll detect the language and translate it for you!</p>
            <p>Try typing "Hello, how are you?" or "Bonjour, comment allez-vous?"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌍 TALKI Translation Chatbot | Powered by Hugging Face Transformers | Built with Streamlit ❤️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()