# chunk_and_index.py

import os
import json
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# --------------------------------------------
# 1. Cấu hình kết nối Redis Stack và embedder SBERT
# --------------------------------------------

# Nếu Docker Desktop của bạn map port 6380:6379, thì host port phải là 6380
REDIS_HOST = "localhost"
REDIS_PORT = 6380         # <--- Cổng 6380 cho khớp với Docker Desktop
INDEX_NAME = "idx:law"

# Khởi tạo kết nối tới Redis Stack
# (Redis Stack có sẵn module RediSearch và VECTOR HNSW)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Khởi tạo SentenceTransformer
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# --------------------------------------------
# 2. Hàm chia văn bản thành các chunk
# --------------------------------------------
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Chia một đoạn text dài thành các chunk (xấp xỉ chunk_size token mỗi chunk),
    với độ overlap giữa các chunk là overlap token.
    Trả về list các chunk (mỗi chunk dưới dạng string).
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# --------------------------------------------
# 3. Hàm sinh embedding bằng SBERT (dim=384)
# --------------------------------------------
def get_embedding(text: str) -> bytes:
    """
    Sinh embedding (float32, dim=384) từ một đoạn text nhờ SentenceTransformers,
    rồi convert numpy array nhanh sang bytes để lưu vào Redis.
    """
    # encode trả về numpy array dtype=float32, shape=(384,)
    emb: np.ndarray = sbert.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    arr = np.array(emb, dtype=np.float32)
    return arr.tobytes()


# --------------------------------------------
# 4. Tạo index trong Redis Stack (chỉ chạy lần đầu)
# --------------------------------------------
def create_redis_index():
    """
    Tạo index tên idx:law trên Redis Stack để dùng VECTOR HNSW + text + meta.
    Nếu đã tồn tại, bỏ qua.
    """
    try:
        r.execute_command(
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", "lawvec:",
            "SCHEMA",
            "vector", "VECTOR", "HNSW", "6",
              "TYPE", "FLOAT32",
              "DIM", "384",
              "DISTANCE_METRIC", "COSINE",
            "text", "TEXT",
            "meta", "TEXT"
        )
        print("✅ Đã tạo index Redis Stack 'idx:law' thành công.")
    except Exception as e:
        # Nếu index đã tồn tại, sẽ ném exception. Bỏ qua.
        print("⚠️ Index đã tồn tại hoặc có lỗi khi tạo, bỏ qua.")
        #print("Chi tiết lỗi:", e)


# --------------------------------------------
# 5. Đọc file .txt, chunk, embed và index vào Redis
# --------------------------------------------
def index_all_chunks(txt_folder: str):
    """
    Duyệt qua từng file .txt trong txt_folder, chia chunk, lấy embedding, lưu vào Redis.
    - Mỗi chunk lưu dưới key "lawvec:<số thứ tự>", các field:
        * vector: embedding (bytes float32)
        * text  : nội dung chunk (string)
        * meta  : metadata dạng JSON string (ví dụ {"source": "filename", "chunk_id": i})
    """
    chunk_counter = 0

    for filename in sorted(os.listdir(txt_folder)):
        if not filename.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(txt_folder, filename)
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = chunk_text(full_text, chunk_size=400, overlap=50)
        print(f"→ Đang index file '{filename}' với {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            meta = {
                "source": filename.replace(".txt", ""),
                "chunk_id": i
                # Nếu bạn parse được Điều/Khoản từ chunk, có thể gán meta["dieu"], meta["khoan"] ở đây
            }
            meta_str = json.dumps(meta, ensure_ascii=False)

            emb_bytes = get_embedding(chunk)
            key = f"lawvec:{chunk_counter}"

            r.hset(key, mapping={
                "vector": emb_bytes,
                "text": chunk,
                "meta": meta_str
            })

            chunk_counter += 1
            if chunk_counter % 100 == 0:
                print(f"   → Đã index tổng cộng {chunk_counter} chunks...")

    print(f"✅ Hoàn tất: đã index {chunk_counter} chunks vào Redis Stack.")


# --------------------------------------------
# 6. Main: tạo index (nếu cần), rồi index toàn bộ chunk
# --------------------------------------------
if __name__ == "__main__":
    # 6.1) Tạo index (chỉ chạy lần đầu, nếu có rồi thì bỏ qua)
    create_redis_index()

    # 6.2) Index tất cả file .txt trong thư mục "plain_texts"
    TXT_FOLDER = "plain_texts"
    if not os.path.isdir(TXT_FOLDER):
        print(f"🚨 Thư mục '{TXT_FOLDER}' không tồn tại. Vui lòng tạo và đặt file .txt vào đó.")
    else:
        index_all_chunks(TXT_FOLDER)
