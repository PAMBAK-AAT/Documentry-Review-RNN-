import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and fixing invisible text
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Button Override */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(99, 102, 241, 0.3);
        background: linear-gradient(90deg, #4338ca 0%, #4f46e5 100%);
    }

    /* Fixed Text Contrast for Divs */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e293b !important; /* Forces dark text even in dark mode */
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Positive & Negative Example Boxes */
    .example-box-pos {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #14532d !important;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .example-box-neg {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #7f1d1d !important;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ef4444;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        color: #0f172a !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    .metric-card p {
        color: #475569 !important; /* Muted text for subtitles */
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_artifacts():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    # Ensure this file exists in your directory!
    model = load_model('simple_rnn_imdb.h5') 
    return model, word_index, reverse_word_index

model, word_index, reverse_word_index = load_artifacts()

# Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Header Section
st.markdown('<div class="main-header">🎬 IMDB Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Recurrent Neural Networks (RNN)</div>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>ℹ️ About This Tool:</strong> This AI uses a pre-trained RNN model trained on 50,000+ IMDB reviews to classify documentary feedback. Enter a review below to get an instant sentiment score.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Main content area
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown("### ✍️ Enter Your Documentary Review")
    user_input = st.text_area(
        label="Review Text",
        label_visibility="collapsed",
        height=180,
        placeholder="Type or paste your documentary review here...\n\nExample: 'This documentary was an amazing exploration of nature. The cinematography was brilliant...'",
    )

    # Word and character count
    if user_input:
        word_count = len(user_input.split())
        char_count = len(user_input)
        count_col1, count_col2 = st.columns(2)
        with count_col1:
            st.caption(f"📝 Words: **{word_count}**")
        with count_col2:
            st.caption(f"🔤 Characters: **{char_count}**")

    st.markdown("---")

    # Example reviews section
    st.markdown("### 💡 Try an Example")
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        st.markdown("""
        <div class="example-box-pos">
            <strong>👍 Positive Example:</strong><br><br>
            <i>"This documentary was an amazing and brilliant masterpiece. The storytelling was outstanding and I absolutely loved every moment of it. Highly recommend!"</i>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Use Positive Example", key="pos"):
            st.session_state['review_text'] = "This documentary was an amazing and brilliant masterpiece. The storytelling was outstanding and I absolutely loved every moment of it. Highly recommend!"
            st.rerun()

    with example_col2:
        st.markdown("""
        <div class="example-box-neg">
            <strong>👎 Negative Example:</strong><br><br>
            <i>"This was the worst documentary I have ever seen. Terrible pacing, boring narration, and a complete waste of time. Very disappointing and dull."</i>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Use Negative Example", key="neg"):
            st.session_state['review_text'] = "This was the worst documentary I have ever seen. Terrible pacing, boring narration, and a complete waste of time. Very disappointing and dull."
            st.rerun()

    # Handle example injection
    if 'review_text' in st.session_state:
        user_input = st.session_state['review_text']
        del st.session_state['review_text']

    st.markdown("---")

    # Prediction button centered
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button('🔮 Classify Sentiment', use_container_width=True)

    if predict_button:
        if not user_input or user_input.strip() == '':
            st.warning("⚠️ Please enter a review before classifying.")
        else:
            with st.spinner('🔄 Analyzing text patterns...'):
                preprocessed_input = preprocess_text(user_input)
                prediction = model.predict(preprocessed_input)
                prediction_score = prediction[0][0]
                sentiment = 'Positive' if prediction_score > 0.5 else 'Negative'
                confidence = max(prediction_score, 1 - prediction_score)

            st.markdown("---")

            # Results section
            st.markdown("## 📈 Analysis Results")

            # Metrics row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(label="Sentiment", value=sentiment)
            with metric_col2:
                st.metric(label="Raw AI Score", value=f"{prediction_score:.4f}")
            with metric_col3:
                st.metric(label="Confidence", value=f"{confidence:.1%}")

            # Sentiment meter
            st.markdown("### Sentiment Meter")
            st.progress(float(prediction_score))
            meter_col1, meter_col2 = st.columns(2)
            with meter_col1:
                st.caption("← Highly Negative (0.0)")
            with meter_col2:
                st.markdown("<p style='text-align: right; font-size: 0.85rem; color: #64748b;'>Highly Positive (1.0) →</p>", unsafe_allow_html=True)

            # Result alerts
            st.write("")
            if sentiment == 'Positive':
                st.success(f"✅ **Positive Sentiment Detected!** The AI is {confidence:.1%} confident that this review is praising the documentary.")
            else:
                st.error(f"⚠️ **Negative Sentiment Detected!** The AI is {confidence:.1%} confident that this review is critical of the documentary.")

    elif not user_input:
        st.info("👆 Please enter a review above and click **Classify Sentiment**.")

with col_side:
    st.markdown("### 📊 Quick Stats")
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="margin: 0; color: #3b82f6;">🎬</h2>
        <h4 style="margin: 10px 0 5px 0; color: #0f172a !important;">IMDB Dataset</h4>
        <p style="margin: 0; font-size: 0.9rem;">Trained on 50,000+ Reviews</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <h2 style="margin: 0; color: #10b981;">🧠</h2>
        <h4 style="margin: 10px 0 5px 0; color: #0f172a !important;">RNN Model</h4>
        <p style="margin: 0; font-size: 0.9rem;">Recurrent Neural Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <h2 style="margin: 0; color: #8b5cf6;">📏</h2>
        <h4 style="margin: 10px 0 5px 0; color: #0f172a !important;">500 Words</h4>
        <p style="margin: 0; font-size: 0.9rem;">Maximum Sequence Length</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ How It Works")
    st.markdown("---")
    
    st.markdown("""
    **1. Text Processing**
    Your text is converted into lower case and split into individual words.
    
    **2. Tokenization**
    Every known word is mapped to a specific integer ID. Unknown words are tagged with a fallback ID.
    
    **3. Padding**
    The sequence is standardized to exactly 500 words. (Shorter reviews get padded; longer ones get truncated).
    
    **4. Neural Analysis**
    The data flows through the Recurrent Neural Network, which evaluates the sequence of words to gauge overall context and sentiment.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    <p>© 2024 Sentiment Analysis System | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)