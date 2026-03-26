# 📖 Medical RAG Application - Technical Documentation

This document provides a detailed explanation of every step performed in the Medical RAG application that compares CBOW and Skip-Gram word embeddings.

---

## 📋 Table of Contents

1. [Application Overview](#application-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Step-by-Step Flow](#step-by-step-flow)
4. [Core Functions Explained](#core-functions-explained)
5. [CBOW vs Skip-Gram Comparison](#cbow-vs-skipgram-comparison)
6. [Technologies Used](#technologies-used)

---

## Application Overview

This is a **Retrieval-Augmented Generation (RAG)** application that:
1. Accepts medical PDF documents
2. Processes and chunks the text
3. Creates word embeddings using both **CBOW** and **Skip-Gram** methods
4. Retrieves relevant chunks for user queries
5. Generates answers using an LLM
6. Displays side-by-side comparison of both methods

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                        (Streamlit - app.py)                     │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  PDF Upload  │───▶│  Text Input  │───▶│   Display    │      │
│  │   Sidebar    │    │   (Query)    │    │   Results    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
├────────────────────────────────────────────────────────────────┤
│                      PROCESSING LAYER                           │
│                        (utils.py)                               │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   PDF    │──▶│  Text    │──▶│  Chunk   │──▶│ Embedding│    │
│  │  Reader  │   │ Preproc  │   │  Split   │   │  Train   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                                  │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│        ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│        │   CBOW   │   │ Skip-Gram│   │  FAISS   │              │
│        │  Model   │   │  Model   │   │  Index   │              │
│        └──────────┘   └──────────┘   └──────────┘              │
│                                                                  │
├────────────────────────────────────────────────────────────────┤
│                       GENERATION LAYER                          │
│                    (HuggingFace API)                            │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Mistral-7B-Instruct LLM                      │  │
│  │         (Answer Generation from Context)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Flow

### Step 1: PDF Upload and Text Extraction

**File:** `utils.py` → `read_pdf()`

```python
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
```

**What happens:**
- User uploads a PDF through Streamlit's file uploader
- PyPDF library opens the PDF and iterates through each page
- Text is extracted from each page and concatenated
- Returns the complete document text as a single string

**Example:**
```
Input: medical_document.pdf (10 pages)
Output: "The digestive system is responsible for breaking down food..."
```

---

### Step 2: Text Preprocessing

**File:** `utils.py` → `preprocess_text()`

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

**What happens:**
1. **Lowercase conversion** - "The Patient" → "the patient"
2. **Special character removal** - Removes symbols except hyphens and periods
3. **Whitespace normalization** - Multiple spaces become single space

**Why it matters:**
- Standardizes text for consistent embeddings
- "Heart" and "heart" become the same token
- Removes noise while keeping medical abbreviations

---

### Step 3: Text Chunking

**File:** `utils.py` → `chunk_text()`

```python
def chunk_text(text, chunk_size=200, overlap=50):
    text = preprocess_text(text)
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
    
    return chunks
```

**What happens:**
- Text is split into chunks of ~200 words each
- Each chunk overlaps with the next by 50 words
- Minimum chunk size is 20 words (filters out tiny fragments)

**Why overlap?**
- Ensures context isn't lost at chunk boundaries
- Important sentences split between chunks are captured

**Example:**
```
Document: 1000 words
Chunk 1: words 0-200
Chunk 2: words 150-350  (50 word overlap)
Chunk 3: words 300-500
...
Total: ~7 chunks
```

---

### Step 4: Word2Vec Model Training

**File:** `utils.py` → `train_word2vec()`

```python
def train_word2vec(sentences, model_type="cbow", use_cache=True):
    # Tokenize sentences
    tokenized = [word_tokenize(sent.lower()) for sent in sentences]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokenized = [[w for w in sent if w not in stop_words and len(w) > 2] 
                 for sent in tokenized]
    
    # Train model
    sg = 0 if model_type == "cbow" else 1
    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,      # 100-dimensional vectors
        window=5,             # Context window of 5 words
        min_count=1,          # Include words appearing once
        sg=sg,                # 0=CBOW, 1=Skip-Gram
        workers=4,            # Parallel training
        epochs=10             # Training iterations
    )
    
    return model
```

**What happens:**
1. **Tokenization** - Splits text into individual words
2. **Stopword removal** - Removes "the", "is", "and", etc.
3. **Model training** - Learns word relationships

**Key Parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| vector_size | 100 | Each word → 100 numbers |
| window | 5 | Consider 5 words left/right |
| min_count | 1 | Include rare words |
| sg | 0/1 | CBOW or Skip-Gram |
| epochs | 10 | Training iterations |

---

### Step 5: Sentence Embedding Creation

**File:** `utils.py` → `sentence_embedding()`

```python
def sentence_embedding(sentence, model):
    words = word_tokenize(sentence.lower())
    vectors = []
    
    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)
```

**What happens:**
1. Sentence is tokenized into words
2. Each word's vector is retrieved from Word2Vec model
3. All word vectors are averaged to get one sentence vector

**Example:**
```
Sentence: "heart disease symptoms"
word_vectors = [
    heart:    [0.2, 0.5, -0.1, ...],  # 100 dims
    disease:  [0.3, 0.4, 0.2, ...],
    symptoms: [0.1, 0.6, 0.0, ...]
]
sentence_vector = mean(word_vectors) = [0.2, 0.5, 0.03, ...]
```

---

### Step 6: FAISS Index Building

**File:** `utils.py` → `build_faiss()`

```python
def build_faiss(chunks, model):
    # Create embeddings for all chunks
    embeddings = np.array(
        [sentence_embedding(chunk, model) for chunk in chunks]
    ).astype("float32")
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings_normalized)
    
    return index, embeddings
```

**What happens:**
1. Every chunk is converted to a 100-dim vector
2. Vectors are normalized (length = 1)
3. FAISS index is created for fast similarity search

**Why FAISS?**
- Facebook AI Similarity Search
- Optimized for fast nearest neighbor search
- Can search millions of vectors in milliseconds

---

### Step 7: Query Processing & Retrieval

**File:** `utils.py` → `retrieve_chunks()`

```python
def retrieve_chunks(query, chunks, model_type, top_k=3):
    # Train embedding model
    w2v_model = train_word2vec(chunks, model_type)
    
    # Build FAISS index
    index, embeddings = build_faiss(chunks, w2v_model)
    
    # Embed the query
    query_vec = sentence_embedding(query, w2v_model).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    # Search for similar chunks
    distances, I = index.search(query_vec.reshape(1, -1), k=top_k)
    
    # Calculate similarity scores
    similarity_scores = [...]
    
    # Return top chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks, metrics
```

**What happens:**
1. Query is embedded using the same Word2Vec model
2. FAISS searches for top-K most similar chunk vectors
3. Returns chunks ranked by cosine similarity

**Example:**
```
Query: "What causes heart disease?"
Query Vector: [0.3, 0.2, 0.5, ...]

Search Results:
  Chunk 7: "heart disease is caused by..." (similarity: 0.85)
  Chunk 12: "cardiovascular problems include..." (similarity: 0.72)
  Chunk 3: "symptoms of heart conditions..." (similarity: 0.68)
```

---

### Step 8: Answer Generation (LLM)

**File:** `app.py` → `generate_answer()`

```python
def generate_answer(client, context, query):
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Answer based ONLY on the context."
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    
    response = client.chat_completion(
        messages=messages,
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_tokens=400,
        temperature=0.3
    )
    
    return response.choices[0].message.content
```

**What happens:**
1. Retrieved chunks are combined as context
2. System prompt instructs LLM to use only provided context
3. User message contains context + question
4. LLM generates an answer based on the context

**LLM Parameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_tokens | 400 | Max response length |
| temperature | 0.3 | Low = more focused answers |

---

### Step 9: Comparison Display

**File:** `app.py`

The app runs Steps 4-8 **twice** - once for CBOW and once for Skip-Gram:

```python
# CBOW pipeline
cbow_chunks, cbow_metrics = retrieve_chunks(query, chunks, "cbow")
cbow_answer = generate_answer(llm_client, cbow_context, query)

# Skip-Gram pipeline  
sg_chunks, sg_metrics = retrieve_chunks(query, chunks, "skipgram")
sg_answer = generate_answer(llm_client, sg_context, query)
```

**Metrics Displayed:**
| Metric | Description |
|--------|-------------|
| Retrieval Time | Time to embed query and search FAISS |
| Similarity Score | Average cosine similarity of retrieved chunks |
| Answer | LLM-generated response |

---

## CBOW vs Skip-Gram Comparison

### CBOW (Continuous Bag of Words)

```
Context: "patient has high ___"
Prediction: "fever" or "blood pressure"
```

- **Training:** Predicts target word from surrounding context
- **Strengths:** Faster, better for frequent words
- **Weakness:** May miss rare medical terms

### Skip-Gram

```
Target: "inflammation"
Prediction: ["patient", "chronic", "tissue", "treatment"]
```

- **Training:** Predicts context words from target word
- **Strengths:** Better for rare words, captures nuances
- **Weakness:** Slower training

### When Each Performs Better

| Scenario | Better Model |
|----------|--------------|
| Common terms (heart, blood) | CBOW |
| Rare diseases/drugs | Skip-Gram |
| Large documents | CBOW |
| Small/specialized corpus | Skip-Gram |

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web UI framework |
| **PyPDF** | PDF text extraction |
| **NLTK** | Tokenization, stopwords |
| **Gensim** | Word2Vec implementation |
| **FAISS** | Fast similarity search |
| **HuggingFace** | LLM API (Mistral-7B) |
| **NumPy** | Vector operations |

---

## File Structure

```
rag_app/
├── app.py              # Main Streamlit application
│   ├── UI Components   # Header, sidebar, forms
│   ├── Session State   # Chat history, PDF cache
│   ├── LLM Setup       # HuggingFace client
│   └── Display Logic   # Results comparison
│
├── utils.py            # Core processing functions
│   ├── read_pdf()      # PDF extraction
│   ├── chunk_text()    # Text splitting
│   ├── train_word2vec()# Embedding training
│   ├── build_faiss()   # Index creation
│   └── retrieve_chunks() # Similarity search
│
├── requirements.txt    # Dependencies
├── .env.example        # API token template
└── README.md           # Project documentation
```

---

## Summary Flow

```
1. User uploads PDF
        ↓
2. Extract text from PDF
        ↓
3. Split into overlapping chunks
        ↓
4. Train CBOW model    &    Train Skip-Gram model
        ↓                          ↓
5. Build FAISS index   &    Build FAISS index
        ↓                          ↓
6. User asks question
        ↓
7. Embed query with CBOW  &  Embed query with Skip-Gram
        ↓                          ↓
8. Search similar chunks  &  Search similar chunks
        ↓                          ↓
9. Generate answer (LLM)  &  Generate answer (LLM)
        ↓                          ↓
10. Display side-by-side comparison with metrics
```

---

*Documentation generated for Medical RAG Comparison Tool*
