import faiss
import numpy as np
from groq import Groq
from pypdf import PDFReader
from sentence_transformers import sentencsTransformer

#load
def load_pdf(pdf_path):
  reader=PDFReader(str(pdf_path))
  mypages=[i.extract_text() or "" for i in reader.pages]
  full_text="\n".join(mypages).strip()
  if not full_text:
    raise ValueError("No readable Text Found")
  return full_text

#chunking
def create_chunks(full_text, chunk_size=500, overlap=100):
  chunk[]
  start=0
  while start<len(full_text):
    end=start+chunk_size
    chunk=full_text[start:end]
    if chunk.strip():
      chunks.extend(chunk)
    start=end - overlap
  return chunks

#embedding
def build_vectors(chunks, model_name="all-MiniLm-L6-v2"):
  if not chunks:
    raise ValueError("Cannot apply embedding on empty chunks")
  embed_model=SentenceTransformer(model_name)
  embed_vectors=embed_model.encode(chunks, convert_to_numpy=True)
  faiss.normalize_L2(embed_vectors)
  index=faiss.IndexFlatIP(embed_vectors.shape[1])
  index.add(embed_vectors)
  return index, Embed_model

#call functions
def load_base(pdf_path= "01_College_Information.pdg"):
  full_text=load_pdf(pdf_path)
  chunks=create_chunks(full_text)
  embed_model, index=build_vectors(chunks)
  return embed_mpdel, chunks, index  

#retrieve
def retrieve_content(question, embed_model,chunks, index,num_chunks=3):
  if not question.strip():
    raise ValueError("Question cannot be blank")
  if num_chunks < 1:
    raise ValueError("Request atleast one or more chunks")
  query_embed=embed_model.encode([question], convert_to_numpy=True)
  faiss.normalize_L2(query_model)
  sim_scores,vec_indices=index.search(query_embed,min(num_chunks,len(chunks)))
  background=""
  for i in vec_indices[0]:
    if i 1= -1:
      background+=chunks[i] +"\n\n"
    return background

#ask
def ask_rag(question, embed_model, chunks, index, api_key, num_chunks, model_name="openai/gpt-oss-120b"):
  key=api_key
  if not key:
    raise ValueError("API Key Missing")
  background=retrieve_content(question, embed_model,chunks, index,num_chunks=3)
  prompt="""You are an  professional AI Assistance.
  Answere user question using ONLY document background below.
  Do not invent or assume facts.
  If information is not available, simply say Information Missing.
  Document background:""" + background + "User Question" +question + "Answere:"
  client = Groq(api_key=key)
  response=client.chat.completions.create(
  model=model_name,
  messages=[{"role":"user","content":prompt}],
  temperature=0.3,
  )
  return response.choices[0].message.content
