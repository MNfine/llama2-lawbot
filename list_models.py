# list_models.py

import os
import google.generativeai as genai

def list_available_models():
    """
    Hàm này sẽ liệt kê hết các model hiện có trên project của bạn.
    Sau khi biết tên đúng của model (vd: 'models/gemini-1.5-pro' hoặc 'models/gemini-2.0-flash'),
    bạn mới dùng vào phần generateContent.
    """
    API_KEY = os.environ.get("GOOGLE_API_KEY", None)
    if API_KEY is None:
        raise RuntimeError("⚠️ Vui lòng set biến môi trường GOOGLE_API_KEY trước khi chạy.")
    # Cấu hình API key
    genai.configure(api_key=API_KEY)

    # Lấy list models
    models = genai.list_models()
    return models

if __name__ == "__main__":
    models = list_available_models()
    print("== Danh sách các model có sẵn ==")
    for m in models:
        # Mỗi phần tử 'm' là một object Model, dùng dot notation
        print(f"- {m.name}\tsupports: {getattr(m, 'supported_methods', None)}")
