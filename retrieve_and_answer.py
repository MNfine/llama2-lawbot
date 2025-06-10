# retrieve_and_answer.py

import os
import json
import redis
import numpy as np
import torch
import re
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

load_dotenv()

# --------------------------------------------
# 1. CẤU HÌNH CHUNG
# --------------------------------------------
# Redis Stack đang chạy Docker Desktop trên port 6380 
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
INDEX_NAME = "idx:law"

# Load Hugging Face token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Số chunk lấy về (top_k) và ngưỡng similarity
TOP_K = 3  # Giảm số chunk lấy về để prompt ngắn hơn
SIMILARITY_THRESHOLD = 0.6

# LLaMA-2-7B (4-bit QLoRA)
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
ADAPTER_PATH   = "qlora_lawbot_output"

# --------------------------------------------
# 2. KẾT NỐI REDIS và SBERT embedder (dim=384)
# --------------------------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") # Model phù hợp cho tiếng Việt


def get_query_embedding(text: str) -> bytes:
    """
    Tạo embedding float32 (dim=384) cho input text,
    rồi convert sang bytes để Redis VECTOR hiểu.
    """
    emb: np.ndarray = sbert.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    arr = np.array(emb, dtype=np.float32)  # float32 array, shape=(384,)
    return arr.tobytes()


# --------------------------------------------
# 3. HÀM RETRIEVE TOP-K CHUNKS TỪ REDIS
# --------------------------------------------
def retrieve_similar_chunks(query: str, top_k: int = TOP_K) -> list:
    """
    Tính embedding của query, tìm top_k chunk gần nhất.
    Trả về list các dict: [{"text": <chunk_text>,
                             "meta": <meta_dict>,
                             "score": <float>}, ...]
    Nếu không tìm thấy hoặc index rỗng, trả về list rỗng.
    """
    q_emb = get_query_embedding(query)

    try:
        raw = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            f"*=>[KNN {top_k} @vector $vec AS vector_score]",
            "PARAMS", 2, "vec", q_emb,
            "RETURN", 3, "text", "meta", "vector_score",
            "DIALECT", 2
        )
    except Exception as e:
        print(f"Lỗi khi FT.SEARCH: {e}")
        return []

    # Nếu raw is None hoặc chỉ có count = 0
    if not raw or len(raw) <= 1:
        return []

    total_hits = int(raw[0])
    print(f"Tìm thấy {total_hits} chunk phù hợp.")

    results = []
    # raw: [total_hits, doc1_id, [field1, value1, field2, value2, ...], doc2_id, [...], ...]
    for i in range(1, len(raw), 2):
        doc_id = raw[i]
        props = raw[i + 1]
        if not isinstance(props, list):
            continue

        props_dict = {}
        # Chuyển mảng [field1, value1, field2, value2, ...] thành dict
        for j in range(0, len(props), 2):
            key = props[j].decode("utf-8") if isinstance(props[j], bytes) else str(props[j])
            val = props[j + 1]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            props_dict[key] = val

        text = props_dict.get("text", "")
        meta_str = props_dict.get("meta", "{}")
        try:
            meta = json.loads(meta_str)
        except:
            meta = {"raw_meta": meta_str}

        try:
            score = float(props_dict.get("vector_score", 0.0))
        except:
            score = 0.0

        # In điểm chunk để kiểm tra
        print(f"[DEBUG] Chunk score: {score:.4f} | Source: {meta.get('source', '')}")

        results.append({"text": text, "meta": meta, "score": score})

    return results

# --------------------------------------------
# 4. CÀI ĐẶT LLaMA-2-QLoRA (4-bit) ĐỂ GENERATE
# --------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("→ Load tokenizer và base LLaMA-2-7B (4-bit)…")
llama_tokenizer = LlamaTokenizer.from_pretrained(
    BASE_MODEL_ID,
    use_fast=False,
    trust_remote_code=True,
    token=HF_TOKEN
)
if llama_tokenizer.pad_token_id is None:
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

# 4.4. Tùy vào GPU/CPU mà load model khác nhau
if torch.cuda.is_available():
    print("→ Phát hiện GPU, sẽ load model 4-bit và offload xuống offload_dir nếu cần …")
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",             # HF tự động phân phối weights GPU/CPU
        offload_folder="offload_dir",   # Bật offload (ghi tạm xuống ổ cứng khi GPU không đủ)
        offload_state_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=HF_TOKEN
    )
else:
    print("→ Không tìm thấy GPU, sẽ load model lên CPU (không quantization) …")
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float16,     # CPU vẫn dùng float16, nhưng chú ý RAM
        trust_remote_code=True,
        token=HF_TOKEN
    )

print("→ Load LoRA adapter từ", ADAPTER_PATH)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=False)
model.eval()


def llama_generate(prompt: str, max_new_tokens: int = 256) -> str:
    # Sinh câu trả lời từ prompt bằng LLaMA-2-QLoRA (4-bit).
    # Trả về phần mới generate (không bao gồm prompt).
    inputs = llama_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            eos_token_id=llama_tokenizer.eos_token_id,
            pad_token_id=llama_tokenizer.pad_token_id
        )
    # Bỏ token của prompt, chỉ lấy phần trả lời
    generated = llama_tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated.strip()

# --------------------------------------------
# 5. HÀM CALL GEMINI (fallback khi RAG không đủ tốt)
# --------------------------------------------
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

def gemini_answer(query: str) -> str:
    """
    Nếu RAG không tìm được chunk đủ tốt, fallback sang gọi Gemini.
    """
    gm = genai.GenerativeModel("models/gemini-2.0-flash")
    resp = gm.generate_content(query)
    return resp.text.strip()


# --------------------------------------------
# 6. XỬ LÝ CÂU HỎI TRẮC NGHIỆM (MCQ)
# --------------------------------------------
def parse_mcq(query: str):
    """
    Kiểm tra xem query có chứa pattern A. … B. … C. … D. …
    nếu có, tách thành (stem, options_dict). 
    Ngược lại, trả về (query, None).
    """
    # Regex tìm A. … (có thể viết trong nhiều dòng) tới D.
    pattern = (
        r"(.*?)(?:\n\s*(?:A\.|A\))\s*(?P<A>.+?))"
        r"(?:\n\s*(?:B\.|B\))\s*(?P<B>.+?))"
        r"(?:\n\s*(?:C\.|C\))\s*(?P<C>.+?))"
        r"(?:\n\s*(?:D\.|D\))\s*(?P<D>.+?))"
    )
    m = re.search(pattern, query, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return query, None

    stem = m.group(1).strip()
    options = {
        "A": m.group("A").strip(),
        "B": m.group("B").strip(),
        "C": m.group("C").strip(),
        "D": m.group("D").strip()
    }
    return stem, options


def answer_mcq_with_context(stem: str, options: dict, chunks: list) -> str:
    """
    Build prompt RAG cho MCQ, rồi gọi LLaMA để chọn đáp án.
    - chunks đã bao gồm text + meta + score.
    """
    # 1. Build phần context (cùng meta nếu có)
    context = ""
    for idx, c in enumerate(chunks):
        meta = c["meta"]
        source = meta.get("source", "Unknown")
        dieu   = meta.get("dieu", "")
        khoan  = meta.get("khoan", "")
        header = f"[Đoạn {idx+1} – {source}"
        if dieu:
            header += f", Điều {dieu}"
        if khoan:
            header += f", Khoản {khoan}"
        header += "]\n"
        context += header + c["text"] + "\n\n"

    # 2. Build phần phương án A/B/C/D
    opts_text = ""
    for key, val in options.items():
        opts_text += f"{key}. {val}\n"

    # 3. Xây prompt MCQ
    prompt = (
        "Bạn là trợ lý hiểu biết sâu về pháp luật Việt Nam.\n"
        "Dưới đây là các đoạn văn bản luật liên quan (có kèm metadata nếu có):\n\n"
        f"{context}"
        f"Hỏi (MCQ): {stem}\n"
        "Phương án:\n" + opts_text + "\n"
        "Hãy chọn 1 trong 4 phương án (A/B/C/D) dựa vào văn bản luật ở trên. "
        "Chỉ trả lời ký tự (A, B, C hoặc D). Nếu có thể, giải thích ngắn gọn.\n"
    )
    
    return llama_generate(prompt)  


# --------------------------------------------
# 7. HÀM KẾT HỢP TRẢ LỜI (RAG + MCQ + fallback Gemini)
# --------------------------------------------
def answer_with_context(query: str) -> str:
    """
    - Nếu query chứa MCQ → tách stem/options, retrieve chunks, build MCQ prompt.
    - Nếu không MCQ → retrieve chunks, build RAG prompt.
    - Nếu không có chunk đủ tốt → fallback sang Gemini.
    """
    # 1. Thử parse MCQ
    stem, options = parse_mcq(query)

    # 2. Retrieve top_k chunks
    chunks = retrieve_similar_chunks(query, top_k=TOP_K)

    # 3. Nếu không tìm được chunk hoặc chunk đầu < threshold → fallback Gemini
    if not chunks or chunks[0]["score"] < SIMILARITY_THRESHOLD:
        return "[Gemini] " + gemini_answer(query)

    # 4. Nếu là MCQ
    if options:
        # Lọc chunks có score >= threshold
        selected = [c for c in chunks if c["score"] >= SIMILARITY_THRESHOLD]
        if not selected:
            return "[Gemini] " + gemini_answer(query)
        return "[LawBot (MCQ)] " + answer_mcq_with_context(stem, options, selected)

    # 5. Bình thường (không phải MCQ) → build RAG prompt
    selected = [c for c in chunks if c["score"] >= SIMILARITY_THRESHOLD]
    if not selected:
        return "[Gemini] " + gemini_answer(query)

    # Build prompt RAG
    context = ""
    for idx, c in enumerate(selected):
        meta = c["meta"]
        source = meta.get("source", "Unknown")
        dieu   = meta.get("dieu", "")
        khoan  = meta.get("khoan", "")
        header = f"[Đoạn {idx+1} – {source}"
        if dieu:
            header += f", Điều {dieu}"
        if khoan:
            header += f", Khoản {khoan}"
        header += "]\n"
        context += header + c["text"] + "\n\n"

    prompt = (
        "Bạn là trợ lý hiểu biết sâu về pháp luật Việt Nam.\n"
        "Các đoạn văn bản luật liên quan (có kèm metadata nếu có):\n\n"
        f"{context}"
        f"Hỏi: {query}\n"
        "Trả lời ngắn gọn, rõ ràng, trích dẫn chính xác Điều, Khoản nếu có."
    )

    return llama_generate(prompt) 

# --------------------------------------------
# 8. KIỂM THỬ TRỰC TIẾP KHI CHẠY FILE NÀY
# --------------------------------------------
if __name__ == "__main__":
    test_mcq = """
    Theo Nghị định 152/2024/NĐ-CP, bạn sẽ bị phạt tiền khi:
    A. Kinh doanh bất động sản chưa được cấp phép.
    B. Không kê khai thuế khi bán bất động sản.
    C. Cho thuê nhà không hợp đồng.
    D. Không đăng ký giấy phép xây dựng.
    """
    print("=== Test MCQ ===")
    print(answer_with_context(test_mcq))
    print("\n=== Test tự luận ===")
    test_query = "Khi nào tôi bị phạt tiền khi kinh doanh bất động sản chưa cấp phép?"
    print(answer_with_context(test_query))
