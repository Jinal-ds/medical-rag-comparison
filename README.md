# 🧬 Medical RAG: CBOW vs Skip-Gram

A Streamlit-based RAG (Retrieval-Augmented Generation) application that compares **CBOW** and **Skip-Gram** word embedding methods for medical document question answering.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jinal-ds-medical-rag-comparison-app-xrtfpe.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)

🔗 **[Try the Live Demo](https://jinal-ds-medical-rag-comparison-app-xrtfpe.streamlit.app/)**


## 🎯 Features

- **PDF Upload**: Upload any medical document for Q&A
- **Dual Embedding Comparison**: See answers from both CBOW and Skip-Gram side-by-side
- **Performance Metrics**: Compare retrieval time and similarity scores
- **Chat History**: Track your conversation with the document
- **Modern UI**: Beautiful gradient-based dark theme

## 🔬 How It Works

1. **Upload** a medical PDF document
2. **Ask** questions about the content
3. **Compare** results from CBOW vs Skip-Gram embeddings

### CBOW vs Skip-Gram

| Feature | CBOW | Skip-Gram |
|---------|------|-----------|
| Approach | Predicts target from context | Predicts context from target |
| Speed | Faster training | Slower training |
| Best For | Common words | Rare words |

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-comparison.git
cd medical-rag-comparison
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your HuggingFace API token
```

### 4. Run the app
```bash
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- HuggingFace API Token (free at [huggingface.co](https://huggingface.co/settings/tokens))

## 🛠️ Tech Stack

- **Streamlit** - Web interface
- **Gensim** - Word2Vec embeddings (CBOW & Skip-Gram)
- **FAISS** - Fast similarity search
- **HuggingFace** - LLM for answer generation
- **PyPDF** - PDF text extraction

## 📁 Project Structure

```
rag_app/
├── app.py           # Main Streamlit application
├── utils.py         # Utility functions (embeddings, retrieval)
├── requirements.txt # Python dependencies
├── .env.example     # Environment variables template
└── README.md        # This file
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
