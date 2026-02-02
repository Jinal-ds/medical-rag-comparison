import nltk
import numpy as np
import time
import hashlib
from gensim.models import Word2Vec
from pypdf import PdfReader
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Cache for Word2Vec models
_model_cache = {}

# -------------------------
# PDF Text Extraction
# -------------------------
def read_pdf(file):
    """Extract text from uploaded PDF file."""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

# -------------------------
# Text Preprocessing
# -------------------------
def preprocess_text(text):
    """Clean and preprocess medical text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep medical abbreviations
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------
# Chunking
# -------------------------
def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks for better context."""
    text = preprocess_text(text)
    words = text.split()
    chunks = []
    
    if len(words) < chunk_size:
        return [text] if text else []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 20:  # Minimum chunk size
            chunks.append(chunk)
    
    return chunks

# -------------------------
# Generate Cache Key
# -------------------------
def _get_cache_key(chunks, model_type):
    """Generate a unique cache key based on content and model type."""
    content = "".join(chunks[:5])  # Use first 5 chunks for key
    content_hash = hashlib.md5(content.encode()).hexdigest()[:10]
    return f"{model_type}_{content_hash}"

# -------------------------
# Train Word2Vec with Caching
# -------------------------
def train_word2vec(sentences, model_type="cbow", use_cache=True):
    """
    Train Word2Vec model with optional caching.
    
    Args:
        sentences: List of text chunks
        model_type: "cbow" or "skipgram"
        use_cache: Whether to use cached model if available
    
    Returns:
        Trained Word2Vec model
    """
    cache_key = _get_cache_key(sentences, model_type)
    
    # Return cached model if available
    if use_cache and cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Tokenize sentences
    tokenized = [word_tokenize(sent.lower()) for sent in sentences]
    
    # Remove stopwords for better embeddings
    stop_words = set(stopwords.words('english'))
    tokenized = [[w for w in sent if w not in stop_words and len(w) > 2] 
                 for sent in tokenized]
    
    # Set skip-gram flag
    sg = 0 if model_type == "cbow" else 1
    
    # Train model with optimized parameters
    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=1,
        sg=sg,
        workers=4,
        epochs=10
    )
    
    # Cache the model
    if use_cache:
        _model_cache[cache_key] = model
    
    return model

# -------------------------
# Sentence Embedding (Average)
# -------------------------
def sentence_embedding(sentence, model, use_tfidf=False, tfidf_weights=None):
    """
    Create sentence embedding by averaging word vectors.
    
    Args:
        sentence: Input text
        model: Word2Vec model
        use_tfidf: Whether to use TF-IDF weighting
        tfidf_weights: Pre-computed TF-IDF weights
    
    Returns:
        Sentence embedding vector
    """
    words = word_tokenize(sentence.lower())
    vectors = []
    weights = []
    
    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
            if use_tfidf and tfidf_weights and w in tfidf_weights:
                weights.append(tfidf_weights[w])
            else:
                weights.append(1.0)
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    if use_tfidf and weights:
        # Weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return np.average(vectors, axis=0, weights=weights)
    
    return np.mean(vectors, axis=0)

# -------------------------
# Build FAISS Index
# -------------------------
def build_faiss(chunks, model):
    """Build FAISS index for fast similarity search."""
    embeddings = np.array(
        [sentence_embedding(chunk, model) for chunk in chunks]
    ).astype("float32")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms
    
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine
    index.add(embeddings_normalized)
    
    return index, embeddings

# -------------------------
# Calculate Similarity Scores
# -------------------------
def calculate_similarity_scores(query_embedding, chunk_embeddings, top_k_indices):
    """Calculate cosine similarity scores for retrieved chunks."""
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    
    scores = []
    for idx in top_k_indices:
        chunk_emb = chunk_embeddings[idx]
        chunk_norm = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-10)
        similarity = np.dot(query_norm, chunk_norm)
        scores.append(float(similarity))
    
    return scores

# -------------------------
# Get Similar Medical Terms
# -------------------------
def get_similar_terms(model, word, topn=5):
    """Get most similar words for a given term."""
    try:
        if word.lower() in model.wv:
            similar = model.wv.most_similar(word.lower(), topn=topn)
            return similar
    except Exception:
        pass
    return []

# -------------------------
# RAG Answer with Metrics
# -------------------------
def rag_answer(query, chunks, model_type, llm, top_k=3):
    """
    Generate RAG answer with performance metrics.
    
    Args:
        query: User question
        chunks: Document chunks
        model_type: "cbow" or "skipgram"
        llm: Language model for generation
        top_k: Number of chunks to retrieve
    
    Returns:
        tuple: (answer, retrieved_chunks, metrics_dict)
    """
    metrics = {
        "model_type": model_type.upper(),
        "embedding_time": 0,
        "retrieval_time": 0,
        "generation_time": 0,
        "total_time": 0,
        "similarity_scores": [],
        "vocab_size": 0
    }
    
    total_start = time.time()
    
    # Train embedding model
    embed_start = time.time()
    w2v_model = train_word2vec(chunks, model_type)
    metrics["vocab_size"] = len(w2v_model.wv)
    metrics["embedding_time"] = time.time() - embed_start
    
    # Build FAISS index
    retrieval_start = time.time()
    index, embeddings = build_faiss(chunks, w2v_model)
    
    # Query embedding
    query_vec = sentence_embedding(query, w2v_model).astype("float32")
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    
    # Retrieve top-k chunks
    distances, I = index.search(query_vec.reshape(1, -1), k=min(top_k, len(chunks)))
    
    metrics["retrieval_time"] = time.time() - retrieval_start
    
    # Calculate similarity scores
    metrics["similarity_scores"] = calculate_similarity_scores(
        query_vec, embeddings, I[0]
    )
    
    # Get retrieved chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join([f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    # Generate answer
    gen_start = time.time()
    
    prompt = f"""You are a helpful medical assistant. Answer the question based ONLY on the provided context. 
If the context doesn't contain enough information to answer, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, accurate, and detailed answer based on the context above. Use medical terminology appropriately."""

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
    
    metrics["generation_time"] = time.time() - gen_start
    metrics["total_time"] = time.time() - total_start
    
    return answer, retrieved_chunks, metrics

# -------------------------
# Clear Model Cache
# -------------------------
def clear_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    _model_cache = {}

# -------------------------
# Retrieve Chunks (without LLM)
# -------------------------
def retrieve_chunks(query, chunks, model_type, top_k=3):
    """
    Retrieve relevant chunks using Word2Vec embeddings.
    
    Args:
        query: User question
        chunks: Document chunks
        model_type: "cbow" or "skipgram"
        top_k: Number of chunks to retrieve
    
    Returns:
        tuple: (retrieved_chunks, metrics_dict)
    """
    metrics = {
        "model_type": model_type.upper(),
        "embedding_time": 0,
        "retrieval_time": 0,
        "similarity_scores": [],
        "vocab_size": 0
    }
    
    total_start = time.time()
    
    try:
        # Train embedding model
        embed_start = time.time()
        w2v_model = train_word2vec(chunks, model_type)
        metrics["vocab_size"] = len(w2v_model.wv)
        metrics["embedding_time"] = time.time() - embed_start
        
        # Build FAISS index
        retrieval_start = time.time()
        index, embeddings = build_faiss(chunks, w2v_model)
        
        # Query embedding
        query_vec = sentence_embedding(query, w2v_model).astype("float32")
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        # Retrieve top-k chunks
        k = min(top_k, len(chunks))
        distances, I = index.search(query_vec.reshape(1, -1), k=k)
        
        metrics["retrieval_time"] = time.time() - retrieval_start
        
        # Calculate similarity scores
        metrics["similarity_scores"] = calculate_similarity_scores(
            query_vec, embeddings, I[0]
        )
        
        # Get retrieved chunks
        retrieved_chunks = [chunks[i] for i in I[0]]
        
        return retrieved_chunks, metrics
        
    except Exception as e:
        metrics["error"] = str(e)
        return [], metrics

