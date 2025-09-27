# 🎭 Emotions & Sentiments Classifier

A comprehensive Natural Language Processing (NLP) web application that performs both sentiment analysis and emotion classification on text using state-of-the-art transformer models from Hugging Face.

## 🌟 Features

- **Dual Classification**: Simultaneous sentiment analysis and emotion detection
- **High Performance**: Utilizes DistilBERT models for fast and accurate predictions
- **GPU Acceleration**: CUDA support for faster inference
- **Interactive Web Interface**: Built with Gradio for easy user interaction
- **Comprehensive Evaluation**: Detailed performance metrics and analysis
- **Batch Processing**: Efficient handling of large text datasets
- **Real-time Demo**: Live web interface with shareable links

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

**Open the notebook in Google Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmerraza12/emotions-sentiments-classifier/blob/main/main.ipynb)

Simply click the badge above to open the notebook directly in Google Colab with all dependencies pre-installed and GPU support available.

**Colab Benefits:**
- ✅ No local installation required
- ✅ Free GPU access (Tesla T4)
- ✅ All dependencies pre-installed
- ✅ Easy sharing and collaboration
- ✅ Automatic environment setup

### Option 2: Local Installation

#### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, for faster inference)
- Required packages (see installation below)

#### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ahmerraza12/emotions-sentiments-classifier.git
cd emotions-sentiments-classifier
```

2. Install dependencies:
```bash
pip install transformers datasets evaluate scikit-learn pandas tqdm gradio torch
```

3. Run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

### Quick Demo

For a quick demonstration, run the Gradio interface:

```python
import gradio as gr
from transformers import pipeline

# Load models
sentiment_pipe = pipeline("sentiment-analysis", 
                         model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipe = pipeline("text-classification", 
                       model="bhadresh-savani/distilbert-base-uncased-emotion", 
                       return_all_scores=True)

def analyze(text):
    s = sentiment_pipe(text)[0]
    e = emotion_pipe(text)[0]
    top3 = sorted(e, key=lambda x: x['score'], reverse=True)[:3]
    top3 = [(d['label'], float(d['score'])) for d in top3]
    return f"{s['label']} ({s['score']:.2f})", top3

iface = gr.Interface(fn=analyze,
                     inputs=gr.Textbox(lines=4, placeholder="Type some text..."),
                     outputs=[gr.Textbox(label="Sentiment"), gr.JSON(label="Top emotions")],
                     title="Sentiment & Emotion Demo")
iface.launch(share=True)
```

## 📊 Models Used

### Sentiment Analysis
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Task**: Binary sentiment classification (Positive/Negative)
- **Performance**: 87.6% accuracy on IMDB test set
- **Dataset**: Stanford Sentiment Treebank (SST-2)

### Emotion Classification
- **Model**: `bhadresh-savani/distilbert-base-uncased-emotion`
- **Task**: Multi-label emotion classification
- **Labels**: 28 emotion categories including joy, sadness, anger, fear, surprise, love, etc.
- **Dataset**: GoEmotions dataset

## 🎯 Performance Metrics

### Sentiment Analysis Results
```
Accuracy: 87.6%
              precision    recall  f1-score   support
         NEG       0.86      0.90      0.88       254
         POS       0.89      0.85      0.87       246
```

### Emotion Classification Results
- **Micro-F1 Score**: 0.08 (threshold 0.3)
- **Subset Accuracy**: 5.75%
- **Top Performing Emotions**: Anger, Joy, Sadness, Love, Surprise

## 🛠️ Technical Details

### Architecture
- **Base Model**: DistilBERT (distilled version of BERT)
- **Framework**: Hugging Face Transformers
- **UI Framework**: Gradio
- **Evaluation**: Scikit-learn metrics
- **Data Processing**: Pandas, NumPy

### Key Features
- **Batch Processing**: Efficient handling of large text datasets with configurable batch sizes
- **Error Handling**: Robust fallback mechanisms for text truncation
- **GPU Support**: Automatic CUDA detection and utilization
- **Memory Optimization**: Smart batching to prevent memory overflow

### Supported Text Lengths
- **Maximum Length**: 512 tokens (configurable)
- **Truncation**: Automatic handling of long texts
- **Batch Size**: Configurable (default: 32 for sentiment, 16 for emotions)

## 📁 Project Structure

```
sentiment-emotion-nlp/
│
├── main.ipynb                    # Main notebook with complete implementation
├── README.md                     # Project overview
├── imdb_sample_preds.csv         # Sample predictions (saved in Colab)
└── goemotions_sample_preds.csv   # Sample predictions (saved in Colab)
```

## 🔧 Usage Examples

### Basic Text Analysis
```python
from transformers import pipeline

# Initialize pipelines
sentiment_pipe = pipeline("sentiment-analysis", 
                         model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipe = pipeline("text-classification", 
                       model="bhadresh-savani/distilbert-base-uncased-emotion", 
                       return_all_scores=True)

# Analyze text
text = "I love this project! It's amazing and makes me so happy."
sentiment = sentiment_pipe(text)
emotions = emotion_pipe(text)

print(f"Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.2f})")
print("Top emotions:")
for emotion in sorted(emotions[0], key=lambda x: x['score'], reverse=True)[:3]:
    print(f"  {emotion['label']}: {emotion['score']:.2f}")
```

### Batch Processing
```python
texts = ["I love this!", "This is terrible.", "I'm so excited!"]
sentiments = batched_pipeline(sentiment_pipe, texts, batch_size=32)
emotions = batched_pipeline(emotion_pipe, texts, batch_size=16)
```


## 📈 Evaluation Results

The project includes comprehensive evaluation on standard datasets:

1. **IMDB Movie Reviews**: 500 sample test set
2. **GoEmotions**: 400 sample test set
3. **Performance Metrics**: Accuracy, F1-score, Precision, Recall
4. **Confusion Matrices**: Detailed per-class performance analysis

## 🚀 Deployment

### Google Colab (Recommended)
- **One-Click Access**: Use the Colab badge above
- **GPU Support**: Free Tesla T4 GPU for faster inference
- **No Setup Required**: All dependencies automatically installed
- **Easy Sharing**: Share notebooks with colleagues instantly

### Local Deployment
```bash
python -m gradio main.py
```

### Cloud Deployment
- **Hugging Face Spaces**: Use `gradio deploy` for permanent hosting
- **Google Colab**: Run the notebook directly in Colab environment
- **Docker**: Containerize for easy deployment


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🙏 Acknowledgments

- **Hugging Face** for the amazing transformers library and pre-trained models
- **Stanford** for the SST-2 dataset
- **Google** for the GoEmotions dataset
- **Gradio** for the intuitive web interface framework

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact [mahmerraza19@outlook.com].

---


