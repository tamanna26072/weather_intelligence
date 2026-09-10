import chromadb
from sentence_transformers import SentenceTransformer
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("monsoon_knowledge")

kb_dir = "docs/knowledge_base"
for fname in os.listdir(kb_dir):
	if fname.endswith(".md"):
		with open(os.path.join(kb_dir, fname), "r", encoding="utf-8") as f:
			text = f.read()
		# simple chunk by paragraph
		chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
		for i, chunk in enumerate(chunks):
			emb = embedder.encode(chunk).tolist()
			collection.add(
				ids=[f"{fname}_{i}"],
				embeddings=[emb],
				documents=[chunk],
				metadatas=[{"source": fname}]
			)
		print(f"{fname}: {len(chunks)} chunks added")

print("Vector store built. Total docs:", collection.count())