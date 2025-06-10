from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from flask_dance.contrib.google import make_google_blueprint, google
from sqlalchemy.exc import IntegrityError
import os
import random
import string
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask import send_from_directory  
import re
import json

# Import các hàm đã được định nghĩa trong retrieve_and_answer.py
from retrieve_and_answer import answer_with_context, retrieve_similar_chunks, parse_mcq
from models import User, db

# Nếu cần OCR
from PIL import Image
import pytesseract

# Nếu cần convert PDF sang ảnh (OCR), cài thư viện pdf2image và poppler-utils
# pip install pdf2image
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# --------------------------------------------
# 1. Thiết lập Flask và cấu hình upload ảnh/PDF
# --------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lawbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # giới hạn 10MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.pdf']
app.config['UPLOAD_PATH'] = 'uploads'

# Email config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# Google OAuth config
app.config['GOOGLE_OAUTH_CLIENT_ID'] = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
app.config['GOOGLE_OAUTH_CLIENT_SECRET'] = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')

mail = Mail(app)

# Thêm dòng này để đăng ký app với SQLAlchemy
db.init_app(app)

# Thiết lập blueprint cho Google OAuth
google_bp = make_google_blueprint(
    client_id=app.config['GOOGLE_OAUTH_CLIENT_ID'],
    client_secret=app.config['GOOGLE_OAUTH_CLIENT_SECRET'],
    scope=["profile", "email"],
    redirect_url="/login/google/authorized"
)
app.register_blueprint(google_bp, url_prefix="/login")

# Thêm đoạn này để khởi tạo Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)

# Thêm filter để định dạng ngày trong Jinja2
@app.template_filter('datetimeformat')
def datetimeformat_filter(value):
    import datetime
    return datetime.datetime.fromtimestamp(value).strftime('%d/%m/%Y %H:%M')

# --------------------------------------------
# 2. Route chính: form hỏi + upload ảnh/PDF
# --------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if not current_user.is_authenticated:
        if session.get('questions_asked', 0) >= 10:
            flash('Vui lòng đăng ký tài khoản để tiếp tục sử dụng')
            return redirect(url_for('register'))
        session['questions_asked'] = session.get('questions_asked', 0) + 1
    elif not current_user.email_verified:
        flash('Vui lòng xác thực email để sử dụng đầy đủ tính năng')
        return redirect(url_for('verify_email', user_id=current_user.id))
    else:
        current_user.questions_asked += 1
        db.session.commit()

    error = None
    query = None
    extracted_text = None
    answer = None

    if request.method == 'POST':
        # Lấy câu hỏi text
        query = request.form.get('query', '').strip()

        # Kiểm tra file upload
        uploaded_file = request.files.get('image')
        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in app.config['UPLOAD_EXTENSIONS']:
                error = "Chỉ cho phép *.jpg, *.jpeg, *.png hoặc *.pdf"
                return render_template("index.html",
                                      error=error,
                                      query=query,
                                      extracted_text=None,
                                      answer=None)
            # Lưu tạm
            temp_path = os.path.join(app.config['UPLOAD_PATH'], filename)
            uploaded_file.save(temp_path)

            # Nếu PDF, convert trang đầu thành ảnh
            if ext == '.pdf':
                if not PDF2IMAGE_AVAILABLE:
                    error = "Thiếu thư viện pdf2image; pip install pdf2image và cài poppler-utils."
                    os.remove(temp_path)
                    return render_template("index.html",
                                          error=error,
                                          query=query,
                                          extracted_text=None,
                                          answer=None)
                try:
                    images = convert_from_path(temp_path, first_page=1, last_page=1)
                    pil_image = images[0]
                except Exception as e:
                    error = f"Lỗi convert PDF sang ảnh: {e}"
                    os.remove(temp_path)
                    return render_template("index.html",
                                          error=error,
                                          query=query,
                                          extracted_text=None,
                                          answer=None)
            else:
                try:
                    pil_image = Image.open(temp_path)
                except Exception as e:
                    error = f"Lỗi mở ảnh: {e}"
                    os.remove(temp_path)
                    return render_template("index.html",
                                          error=error,
                                          query=query,
                                          extracted_text=None,
                                          answer=None)

            # OCR
            try:
                extracted_text = pytesseract.image_to_string(pil_image, lang='vie')
            except Exception as e:
                error = f"Lỗi OCR: {e}"
                os.remove(temp_path)
                return render_template("index.html",
                                      error=error,
                                      query=query,
                                      extracted_text=None,
                                      answer=None)

            # Xóa file tạm
            try:
                os.remove(temp_path)
            except:
                pass

            # Nếu user chưa gõ query, dùng extracted_text làm query
            if not query and extracted_text:
                query = extracted_text.strip()
        if not query:
            error = "Bạn cần nhập câu hỏi hoặc upload ảnh/PDF."
            return render_template("index.html",
                                  error=error,
                                  query=query,
                                  extracted_text=extracted_text,
                                  answer=None)
        try:
            answer = answer_with_context(query)
        except Exception as e:
            answer = f"Đã có lỗi khi xử lý: {e}"

    return render_template("index.html",
                          error=error,
                          query=query,
                          extracted_text=extracted_text,
                          answer=answer)


# --------------------------------------------
# 4. Route phụ (debug): hiển thị top-k chunks
# --------------------------------------------
@app.route('/chunks', methods=['GET'])
def show_chunks():
    example = request.args.get('q', '').strip()
    if not example:
        return "<p>Thêm ?q=… vào URL để xem top-k chunks tương tự.</p>"

    chunks = retrieve_similar_chunks(example, top_k=5)
    html = "<h3>Query:</h3><pre>{}</pre><h3>Top-5 chunks:</h3>".format(example)
    for i, c in enumerate(chunks, start=1):
        try:
            meta = c['meta']
            source = meta.get('source', 'Unknown')
            txt = c['text'].replace('\n', '<br>')
            sc = c['score']
        except:
            source = "Unknown"
            txt = c['text'].replace('\n', '<br>') if 'text' in c else ""
            sc = c.get('score', 0.0)
        html += f"<b>Chunk {i} (score={sc:.4f}, source={source}):</b><br><pre>{txt[:500]}...</pre><hr>"
    return html


# --------------------------------------------
# 5. Authentication routes
# --------------------------------------------
# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        phone = request.form.get('phone')
        
        if User.query.filter_by(email=email).first():
            flash('Email này đã được đăng ký, vui lòng sử dụng email khác')
            return redirect(url_for('register'))
            
        user = User(email=email, username=username, phone=phone)
        user.set_password(password)
        
        # Tạo mã xác thực
        code = ''.join(random.choices(string.digits, k=6))
        user.verification_code = code
        user.code_expiry = datetime.utcnow() + timedelta(minutes=15)
        
        db.session.add(user)
        db.session.commit()
        
        # Gửi email xác thực
        msg = Message(
            'Xác thực tài khoản LawBot',
            sender='noreply@lawbot.com',
            recipients=[email]
        )
        msg.body = f'''
        Xin chào {username},

        Cảm ơn bạn đã đăng ký tài khoản LawBot.
        Mã xác thực của bạn là: {code}
        Mã có hiệu lực trong vòng 15 phút.

        Trân trọng,
        Đội ngũ LawBot
        '''
        mail.send(msg)
        
        flash('Vui lòng kiểm tra email để lấy mã xác thực')
        return redirect(url_for('verify_email', user_id=user.id))
    return render_template("register.html")

# Email verification route
@app.route('/verify-email/<int:user_id>', methods=['GET', 'POST']) 
def verify_email(user_id):
    if request.method == 'POST':
        code = request.form.get('code')
        user = User.query.get(user_id)
        
        if not user or user.verification_code != code:
            flash('Mã xác thực không đúng')
            return redirect(url_for('verify_email', user_id=user_id))
            
        if datetime.utcnow() > user.code_expiry:
            flash('Mã xác thực đã hết hạn')
            return redirect(url_for('verify_email', user_id=user_id))
            
        user.email_verified = True
        user.verification_code = None
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('index'))
    return render_template("verify_email.html")

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email') 
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            if not user.email_verified:
                flash('Vui lòng xác thực email trước')
                return redirect(url_for('verify_email', user_id=user.id))
                
            login_user(user)
            return redirect(url_for('index'))
            
        flash('Email hoặc mật khẩu không đúng')
    return render_template("login.html")

# Google login route
@app.route("/login/google")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Không thể lấy thông tin từ Google.")
        return redirect(url_for("login"))
    info = resp.json()
    email = info.get("email")
    username = info.get("name") or email.split("@")[0]
    user = User.query.filter_by(email=email).first()
    if not user:
        # Tạo user mới nếu chưa có
        user = User(email=email, username=username, email_verified=True)
        try:
            db.session.add(user)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            flash("Có lỗi khi tạo tài khoản Google.")
            return redirect(url_for("login"))
    login_user(user)
    return redirect(url_for("index"))

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --------------------------------------------
# 6. Thêm route giới thiệu luật sư
# --------------------------------------------
@app.route('/lawyer')
def lawyer():
    return render_template("lawyer.html")

# --------------------------------------------
# 7. Route hiển thị tin tức (văn bản pháp luật mới)
# --------------------------------------------
@app.route('/news')
def news():
    plain_texts_dir = os.path.join(os.getcwd(), "plain_texts")
    docs = []

    search_query = request.args.get('q', '').lower().strip()
    selected_fields = request.args.getlist('field')
    selected_doc_types = request.args.getlist('doc_type')
    selected_agencies = request.args.getlist('agency')

    # Map các tên viết tắt và pattern
    doc_type_patterns = {
        "nghị định": r"nghị[- ]định|nđ[-]?cp",
        "thông tư": r"thông[- ]tư|tt[-]btp",
        "quyết định": r"quyết[- ]định|qđ",
        "luật": r"\bluật\b",
        "nghị quyết": r"nghị[- ]quyết",
        "chỉ thị": r"chỉ[- ]thị"
    }

    agency_patterns = {
        "chính phủ": r"chính[- ]phủ|cp\b|nđ[-]?cp",
        "bộ tư pháp": r"bộ[- ]tư[- ]pháp|btp",
        "bộ tài chính": r"bộ[- ]tài[- ]chính|btc", 
        "thủ tướng chính phủ": r"thủ[- ]tướng|ttcp",
        "tòa án nhân dân tối cao": r"tòa[- ]án|tandtc",
        "quốc hội": r"quốc[- ]hội|qh"
        # Thêm các pattern khác...
    }
    
    # Đọc fields.meta (JSON) duy nhất cho toàn bộ plain_texts
    fields_meta_path = os.path.join(plain_texts_dir, "fields.meta")
    if os.path.exists(fields_meta_path):
        with open(fields_meta_path, "r", encoding="utf-8") as f:
            fields_dict = json.load(f)
    else:
        fields_dict = {}

    if os.path.isdir(plain_texts_dir):
        for fname in sorted(os.listdir(plain_texts_dir), reverse=True):
            if fname.lower().endswith(".txt"):
                fpath = os.path.join(plain_texts_dir, fname)
                name = fname.replace(".txt", "").lower()
                should_include = True

                # 1. Kiểm tra từ khóa tìm kiếm trong tên file
                if search_query and search_query not in name:
                    should_include = False
                    continue

                # 2. Kiểm tra loại văn bản bằng regex pattern
                if selected_doc_types:
                    doc_type_matched = False
                    for doc_type in selected_doc_types:
                        pattern = doc_type_patterns.get(doc_type.lower())
                        if pattern and re.search(pattern, name, re.I):
                            doc_type_matched = True
                            break
                    if not doc_type_matched:
                        should_include = False
                        continue

                # 3. Kiểm tra cơ quan ban hành bằng regex pattern
                if selected_agencies:
                    agency_matched = False
                    for agency in selected_agencies:
                        pattern = agency_patterns.get(agency.lower())
                        if pattern and re.search(pattern, name, re.I):
                            agency_matched = True
                            break
                    if not agency_matched:
                        should_include = False
                        continue

                # 4. Chỉ đọc nội dung nếu cần lọc lĩnh vực
                if should_include and selected_fields:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if not any(field.lower() in content for field in selected_fields):
                            should_include = False
                            continue

                # Lấy lĩnh vực từ fields.meta nếu có
                field = fields_dict.get(fname, "")

                # Lọc chính xác theo lĩnh vực nếu có selected_fields
                if selected_fields:
                    # Nếu không có field hoặc field không nằm trong selected_fields thì loại bỏ
                    if not field or field not in selected_fields:
                        should_include = False
                        continue

                if should_include:
                    docs.append({
                        "name": fname.replace(".txt", ""),
                        "filename": fname,
                        "mtime": os.path.getmtime(fpath),
                        "field": field
                    })

    # Sắp xếp theo ngày mới nhất
    docs.sort(key=lambda d: d["mtime"], reverse=True)
    
    return render_template("news.html",
                         docs=docs,
                         search_query=search_query,
                         selected_fields=selected_fields,
                         selected_types=selected_doc_types,
                         selected_agencies=selected_agencies)

@app.route('/plain_texts/<path:filename>')
def download_plain_text(filename):
    return send_from_directory("plain_texts", filename, as_attachment=True)

# --------------------------------------------
# 6. Chạy app trên cổng 5002
# --------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5002, debug=True)
