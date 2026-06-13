from sentence_transformers import SentenceTransformer


def get_sentence_embedding(sentence):
    model = SentenceTransformer('/Models/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return embeddings
