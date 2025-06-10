import redis
import numpy as np
from sentence_transformers import SentenceTransformer

# Cấu hình Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6380
INDEX_NAME = "idx:law"

# Kết nối Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Khởi tạo SentenceTransformer
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding(text: str) -> bytes:
    emb = sbert.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    arr = np.array(emb, dtype=np.float32)
    return arr.tobytes()

# Query test
query = "Khi nào tôi bị phạt tiền khi kinh doanh bất động sản chưa cấp phép?"
q_emb = get_embedding(query)

# Search và in chi tiết kết quả 
raw = r.execute_command(
    "FT.SEARCH", INDEX_NAME,
    f"*=>[KNN 3 @vector $vec]",
    "PARAMS", 2, "vec", q_emb,
    "RETURN", 3, "text", "meta", "vector_score",
    "DIALECT", 2
)

print("=== Raw Response ===")
print(f"Type: {type(raw)}")
print(f"Length: {len(raw)}")
print("\nElements:")
for i, item in enumerate(raw):
    if isinstance(item, bytes):
        try:
            print(f"{i}: {item.decode('utf-8')}")
        except:
            print(f"{i}: <binary data>")
    else:
        print(f"{i}: {item}")
