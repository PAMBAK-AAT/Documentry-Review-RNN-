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

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 50%, #b45309 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(245, 158, 11, 0.3);
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #51cf6620 0%, #3cb37120 100%);
        border: 2px solid #51cf66;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff6b6b20 0%, #ee5a5a20 100%);
        border: 2px solid #ff6b6b;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    .example-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_artifacts():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
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
st.markdown('<div class="main-header">🎬 IMDB Documentary Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Recurrent Neural Network (RNN) | Classify reviews as Positive or Negative</div>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    ℹ️ <strong>About This Tool:</strong> This AI-powered tool uses a pre-trained RNN model trained on the IMDB dataset 
    to classify documentary reviews as positive or negative. Enter or paste a review below to get instant sentiment analysis.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Main content area
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown("### ✍️ Enter Your Documentary Review")
    user_input = st.text_area(
        'Documentary Review',
        height=200,
        placeholder="Type or paste your documentary review here...\n\nExample: 'This documentary was an amazing exploration of nature. The cinematography was brilliant and the storytelling was excellent.'",
        help="Enter a documentary review to classify its sentiment"
    )

    # Word and character count
    if user_input:
        word_count = len(user_input.split())
        char_count = len(user_input)
        count_col1, count_col2 = st.columns(2)
        with count_col1:
            st.caption(f"📝 Words: {word_count}")
        with count_col2:
            st.caption(f"🔤 Characters: {char_count}")

    st.markdown("---")

    # Example reviews section
    st.markdown("### 💡 Try an Example Review")
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        st.markdown("""
        <div class="example-box">
            👍 <strong>Positive Example:</strong><br>
            "This documentary was an amazing and brilliant masterpiece. The storytelling was outstanding and I absolutely loved every moment of it. Highly recommend!"
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Use Positive Example", key="pos"):
            st.session_state['review_text'] = "This documentary was an amazing and brilliant masterpiece. The storytelling was outstanding and I absolutely loved every moment of it. Highly recommend!"
            st.rerun()

    with example_col2:
        st.markdown("""
        <div class="example-box">
            👎 <strong>Negative Example:</strong><br>
            "This was the worst documentary I have ever seen. Terrible pacing, boring narration, and a complete waste of time. Very disappointing and dull."
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Use Negative Example", key="neg"):
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
            st.warning("⚠️ Please enter a documentary review before classifying.")
        else:
            with st.spinner('🔄 Analyzing review sentiment...'):
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
                st.metric(label="Prediction Score", value=f"{prediction_score:.4f}")

            with metric_col3:
                st.metric(label="Confidence", value=f"{confidence:.1%}")

            # Sentiment meter
            st.markdown("### Sentiment Meter")
            st.progress(float(prediction_score))
            meter_col1, meter_col2 = st.columns(2)
            with meter_col1:
                st.caption("← Negative")
            with meter_col2:
                st.markdown("<p style='text-align: right; font-size: 0.85rem; color: #6b7280;'>Positive →</p>", unsafe_allow_html=True)

            # Result card
            if sentiment == 'Positive':
                st.success(f"""
                ### ✅ Positive Sentiment Detected!

                **Prediction Score: {prediction_score:.4f}**

                The review expresses a positive sentiment. Key insights:
                - 🌟 The reviewer likely enjoyed the documentary
                - 👍 Positive language and tone detected
                - 📊 Confidence: {confidence:.1%}
                - 🎬 Likely to recommend to others
                """)
            else:
                st.error(f"""
                ### ⚠️ Negative Sentiment Detected!

                **Prediction Score: {prediction_score:.4f}**

                The review expresses a negative sentiment. Key insights:
                - 😞 The reviewer likely did not enjoy the documentary
                - 👎 Negative language and tone detected
                - 📊 Confidence: {confidence:.1%}
                - 🎬 Unlikely to recommend to others
                """)

    elif not user_input:
        st.info("📝 Please enter a documentary review above and click **Classify Sentiment** to get started.")

with col_side:
    st.markdown("### 📊 Quick Stats")
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin: 0; color: #f59e0b;">🎬</h3>
        <p style="margin: 0; font-weight: 600;">IMDB Dataset</p>
        <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">50,000+ Reviews</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class="metric-card">
        <h3 style="margin: 0; color: #10b981;">🧠</h3>
        <p style="margin: 0; font-weight: 600;">RNN Model</p>
        <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">Simple RNN Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class="metric-card">
        <h3 style="margin: 0; color: #6366f1;">📏</h3>
        <p style="margin: 0; font-weight: 600;">Max Length</p>
        <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">500 Words (Padded/Truncated)</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 🎬 Model Information")
    st.markdown("---")

    st.markdown("""
    ### 🧠 Neural Network Details
    - **Model Type:** Simple RNN
    - **Framework:** TensorFlow/Keras
    - **Dataset:** IMDB Reviews
    - **Vocabulary:** Full IMDB word index
    - **Max Sequence Length:** 500
    - **Output:** Binary Classification
    """)

    st.markdown("---")

    st.markdown("""
    ### 🔧 Preprocessing Pipeline
    1. Text lowercased
    2. Words tokenized (split by space)
    3. Words mapped to integer indices
    4. Sequences padded/truncated to 500
    5. Fed into RNN for prediction
    """)

    st.markdown("---")

    st.markdown("""
    ### 📋 How Scoring Works
    - **Score > 0.5** → Positive sentiment
    - **Score ≤ 0.5** → Negative sentiment
    - Score closer to 1.0 = stronger positive
    - Score closer to 0.0 = stronger negative
    """)

    st.markdown("---")

    st.markdown("""
    ### 📝 How to Use
    1. Enter a documentary review
    2. Click **Classify Sentiment**
    3. View sentiment & confidence
    4. Try example reviews to test
    """)

    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit & TensorFlow*")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6b7280;">
    <p>© 2024 IMDB Sentiment Analysis System | Powered by Recurrent Neural Network</p>
</div>
""", unsafe_allow_html=True)
