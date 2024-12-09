import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="Penerjemah Indonesia-Minangkabau",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to make it look more like Google Translate
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
        .translate-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stTextArea textarea {
            border: none;
            background-color: #f8f9fa;
            font-size: 16px;
            resize: none;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            border-radius: 4px;
            padding: 4px 24px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1557b0;
        }
        .swap-button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 10px 0;
        }
        div[data-testid="stSelectbox"] > div > div {
            background-color: #f8f9fa;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

def load_model(model_path):
    """Load model and tokenizer from local path"""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def translate_text(text, model, tokenizer, max_length=128):
    """Perform translation using the loaded model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        length_penalty=0.6,
        early_stopping=True
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def main():
    # Initialize session state if needed
    if 'translated' not in st.session_state:
        st.session_state.translated = ""

    # Minimal header
    st.markdown("<h1 style='text-align: center; color: #5f6368; font-size: 24px; margin-bottom: 20px;'>Penerjemah Indonesia-Minangkabau</h1>", unsafe_allow_html=True)
    
    # Load models
    @st.cache_resource
    def load_translation_models():
        indo_minang_model, indo_minang_tokenizer = load_model("indonesia-minangkabau")
        minang_indo_model, minang_indo_tokenizer = load_model("minangkabau-indonesia")
        return (indo_minang_model, indo_minang_tokenizer, 
                minang_indo_model, minang_indo_tokenizer)
    
    try:
        (indo_minang_model, indo_minang_tokenizer,
         minang_indo_model, minang_indo_tokenizer) = load_translation_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        models_loaded = False
    
    if models_loaded:
        # Create main container
        with st.container():
            col1, col2 = st.columns([5, 5])
            
            # Source language column
            with col1:
                st.markdown("<p style='color: #5f6368; font-size: 14px;'>Dari:</p>", unsafe_allow_html=True)
                source_lang = st.selectbox(
                    "",
                    ["Indonesia", "Minangkabau"],
                    label_visibility="collapsed"
                )
                
                input_text = st.text_area(
                    "",
                    height=200,
                    placeholder="Ketik atau tempel teks di sini...",
                    label_visibility="collapsed"
                )
                
                # Character count
                if input_text:
                    st.markdown(f"<p style='color: #5f6368; font-size: 12px; text-align: right;'>{len(input_text)}/5000</p>", unsafe_allow_html=True)

            # Target language column
            with col2:
                st.markdown("<p style='color: #5f6368; font-size: 14px;'>Ke:</p>", unsafe_allow_html=True)
                target_lang = "Minangkabau" if source_lang == "Indonesia" else "Indonesia"
                st.markdown(f"<p style='color: #1a73e8; font-size: 14px;'>{target_lang}</p>", unsafe_allow_html=True)
                
                # Translation output area
                st.markdown(
                    f"""<div style='background-color: #f8f9fa; padding: 10px; 
                    border-radius: 4px; min-height: 200px; color: #5f6368;'>
                    {st.session_state.translated if st.session_state.translated else 'Terjemahan akan muncul di sini'}</div>""", 
                    unsafe_allow_html=True
                )

            # Action buttons
            col3, col4 = st.columns([5, 5])
            with col3:
                if st.button("🔄 Terjemahkan", key="translate_button"):
                    if input_text:
                        try:
                            with st.spinner("Menerjemahkan..."):
                                if source_lang == "Indonesia":
                                    translated = translate_text(
                                        input_text, 
                                        indo_minang_model, 
                                        indo_minang_tokenizer
                                    )
                                else:
                                    translated = translate_text(
                                        input_text,
                                        minang_indo_model,
                                        minang_indo_tokenizer
                                    )
                                st.session_state.translated = translated
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error during translation: {str(e)}")
                    else:
                        st.warning("Mohon masukkan teks yang ingin diterjemahkan")
                
            with col4:
                if st.button("📋 Salin", key="copy_translation"):
                    if st.session_state.translated:
                        st.success("Teks telah disalin ke clipboard!")

if __name__ == "__main__":
    main()