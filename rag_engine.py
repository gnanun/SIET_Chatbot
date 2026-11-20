import faiss  # Used for similarity search.
import numpy as np  # Used for numerical operations.
from groq import Groq  # Connects to Groq LLM.
from pypdf import PdfReader  # Reads PDF files.
from sentence_transformers import SentenceTransformer  # Generates text embeddings.


# 1. LOAD PDF
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))  # Opens the PDF file.
    pages = [page.extract_text() or "" for page in reader.pages]  # Extracts text from every page.
    full_text = "\n".join(pages).strip()  # Combines all page texts into one string.

    if not full_text:  # Checks whether PDF contains readable text.
        raise ValueError("No readable text found")

    return full_text  # Returns the extracted PDF text.


# 2. CREATE CHUNKS
def create_chunks(full_text, chunk_size=500, overlap=100):
    chunks = []  # Creates an empty list to store chunks.
    start = 0  # Initializes the starting position.

    while start < len(full_text):  # Loops until the complete text is processed.
        end = start + chunk_size  # Calculates the ending position.
        chunk = full_text[start:end]  # Extracts a portion of text.

        if chunk.strip():  # Checks whether the chunk is not empty.
            chunks.append(chunk)  # Adds the chunk to the list.

        start = end - overlap  # Moves forward while maintaining overlap.

    return chunks  # Returns the list of chunks.


# 3. BUILD VECTORS
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:  # Checks whether chunks are available.
        raise ValueError("Cannot apply embedding on empty chunks")

    embed_model = SentenceTransformer(model_name)  # Loads the embedding model.
    embed_vectors = embed_model.encode(chunks, convert_to_numpy=True)  # Converts chunks into numerical vectors.
    embed_vectors = np.asarray(embed_vectors, dtype="float32")  # Converts vectors into FAISS-compatible format.
    faiss.normalize_L2(embed_vectors)  # Normalizes vectors for cosine similarity.
    index = faiss.IndexFlatIP(embed_vectors.shape[1])  # Creates a FAISS similarity search index.
    index.add(embed_vectors)  # Stores vectors in the index.

    return embed_model, index  # Returns the model and FAISS index.


# 4. LOAD BASE
def load_base(pdf_path="01_College_Information.pdf"):
    full_text = load_pdf(pdf_path)  # Loads and extracts PDF text.
    chunks = create_chunks(full_text)  # Splits text into overlapping chunks.
    embed_model, index = build_vectors(chunks)  # Generates embeddings and builds the index.

    return index, embed_model, chunks  # Returns all required RAG components.


# 5. RETRIEVE CONTENT
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():  # Checks whether the question is empty.
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:  # Checks whether the requested chunk count is valid.
        raise ValueError("Request at least one chunk")

    query_vector = embed_model.encode([question], convert_to_numpy=True)  # Converts question into an embedding.
    query_vector = np.asarray(query_vector, dtype="float32")  # Converts query into float32 format.
    faiss.normalize_L2(query_vector)  # Normalizes the query vector.

    sim_scores, vec_indices = index.search(query_vector, min(num_chunks, len(chunks)))  # Finds the most similar chunks.

    background = ""  # Initializes an empty background string.

    for i in vec_indices[0]:  # Iterates through retrieved chunk indices.
        if i != -1:  # Checks whether the retrieved index is valid.
            background += chunks[i] + "\n\n"  # Adds retrieved text to the background.

    return background  # Returns the relevant document content.

# 6. ASK RAG
def ask_rag(question, embed_model, chunks, index, api_key, num_chunks=3, model_name="openai/gpt-oss-120b"):

    if not api_key:  # Checks whether the API key is available.
        raise ValueError("API Key Missing")

    background = retrieve_content(question, embed_model, chunks, index, num_chunks)  # Retrieves relevant PDF content.

    prompt = f"""You are a professional AI Assistant.
    Answer the user's question using ONLY the document background below.
    Do not invent or assume facts.
    If information is unavailable, say Information Missing.

    Document Background:
    {background}

    User Question:
    {question}

    Answer:
    """  # Creates a prompt using retrieved content and the user question.

    client = Groq(api_key=api_key)  # Creates a Groq API client.

    response = client.chat.completions.create(  # Sends the prompt to the Groq model.
        model=model_name,  # Specifies the language model.
        messages=[{"role": "user", "content": prompt}],  # Passes the prompt as a user message.
        temperature=0.3  # Controls the randomness of the generated answer.
    )

    return response.choices[0].message.content  # Returns the generated answer.
