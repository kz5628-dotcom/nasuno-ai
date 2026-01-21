import streamlit as st
import google.generativeai as genai
import os
import time
import datetime
import qrcode
import json
from PIL import Image
from io import BytesIO
from google.api_core.exceptions import ResourceExhausted

# ==========================================
#  設定エリア
# ==========================================
st.set_page_config(page_title="AI問診 - 那須乃アイ", page_icon="🏥", layout="wide") 

# APIキーの読み込み
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("APIキー設定エラー: .streamlit/secrets.toml (またはCloudのSecrets) を確認してください。")
    st.stop()

# 今日の日付
today_str = datetime.date.today().strftime("%Y/%m/%d")

# ==========================================
#  セッション状態の初期化
# ==========================================
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None 

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# ==========================================
#  関数定義：時間帯ごとの挨拶
# ==========================================
def get_time_based_greeting():
    hour = datetime.datetime.now().hour
    if hour < 10:
        return "おはようございます"
    elif hour < 18:
        return "こんにちは"
    else:
        return "こんばんは"

# ==========================================
#  関数定義：QRコード用のテキスト整形
# ==========================================
def format_text_for_qr(text):
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        if "氏名：" in line and "生)" in line:
            continue
        
        clean_line = line.replace("■ S (Subjective)", "【S】")
        clean_line = clean_line.replace("■ O (Objective)", "【O】")
        clean_line = clean_line.replace("■ 患者希望", "【P】")
        clean_line = clean_line.replace("■ Plan", "【P】")
        
        if "---" in clean_line:
            continue
            
        new_lines.append(clean_line)
    
    formatted_text = "\r\n".join(new_lines)
    
    while "\r\n\r\n\r\n" in formatted_text:
        formatted_text = formatted_text.replace("\r\n\r\n\r\n", "\r\n\r\n")
        
    return formatted_text.strip()

# ==========================================
#  関数定義：画像から個人情報を抽出
# ==========================================
def extract_patient_info(image_data):
    model = genai.GenerativeModel('gemini-2.5-flash')
    img = Image.open(image_data)
    prompt = """
    この身分証（マイナンバーカード等）の画像から、以下の情報を読み取ってください。
    1. 氏名（漢字）
    2. 生年月日（西暦yyyy年mm月dd日形式に変換）
    
    出力は以下のJSON形式のみで行ってください。余計な文章は一切不要です。
    ```json
    {
        "name": "氏名",
        "dob": "yyyy年mm月dd日"
    }
    ```
    """
    try:
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        json_str = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        return data
    except Exception as e:
        st.error(f"読み取りエラー: {e}")
        return None

# ==========================================
#  関数定義：問診AI（チャット）
# ==========================================
def generate_response_with_fallback(chat_history, patient_name, patient_dob):
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
    ]
    
    greeting = get_time_based_greeting()

    DYNAMIC_SYSTEM_PROMPT = f"""
    あなたは整形外科クリニックのAI問診担当「那須乃アイ（なすのあい）」です。
    以下の【問診フロー】に従って、患者と対話し、情報を収集してください。
    **本日の日付は {today_str} です。**

    【重要：患者情報】
    * **氏名:** {patient_name}
    * **生年月日:** {patient_dob}
    この情報は取得済みです。「2. 患者情報」はスキップしてください。

    【問診フロー】
    1. 挨拶＆初診確認：
       ※最初の発言で「{patient_name}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。当院へのご来院は初めてですか？」と挨拶と質問をまとめて行っている状態からスタートします。
       **したがって、ユーザー（患者）からの最初の返答は「初診かどうか」の回答になります。**
    
    2. (スキップ)
    3. (スキップ)
    4. 初診確認の続き：
       もし患者の返答が初診かどうかわからない場合は聞き直す。わかった場合は次に進む。
       
    5. (再診の場合のみ)：医師名の確認
    6. 来院理由：「本日はどういった症状でご来院されましたか？」
    7. 主訴の確認：聞き取った症状を箇条書きで確認
    8. 詳細聴取：発症起点、原因、増悪因子などを詳しく
    9. まとめ確認
    10. 画像検査の希望
    11. 骨粗鬆症検査の希望
    12. 医師希望
    13. 終了案内

    【最終出力フォーマットの厳格なルール】
    会話終了後、以下のSOAP形式で出力してください。Markdownのコードブロックは使用せず、テキストとして出力すること。

    ### 1. 日付記載の絶対ルール
    すべての相対的な日付表現は、**「出力を行った当日（{today_str}）」を基準日**として、以下の形式で具体的な日付に変換して記載すること。
    * **基準:** 本日（{today_str}）
    * **昨日:** 正確な日付で記載（例: yyyy/mm/dd）
    * **2週間前:** 14日前を計算して「yyyy/mm/dd頃」とする
    * **1ヶ月前:** 月単位の場合は「上旬/中旬/下旬」で表現
    * **半年以上前:** 「yyyy/mm月下旬頃」または年単位の経過として記載。
    * **禁止事項:** 「1週間前」「数日前」「昨日」などの相対表現をそのまま出力に残さないこと。これらは必ず日付に変換すること。

    ### 2. 出力テンプレート
    ---
    ■ S (Subjective)
    氏名：{patient_name} ({patient_dob}生)
    主訴：
    #1. (名詞または体言止めで簡潔に記載)
    #2. (名詞または体言止めで簡潔に記載)
    (※各項目の間には空行を入れず詰めること)

    現病歴：
    (発症起点、原因、経過などを記載。日付は必ずyyyy/mm/dd形式)
    (※最終行は必ず以下の3つのうちいずれか1文で締めくくる)
    1. {today_str} 　症状が持続しているため当院を受診
    2. {today_str} 　症状が改善しないため当院を受診
    3. {today_str} 　症状が悪化してきたため当院を受診

    ■ O (Objective)
    (問診で得られた症状の補足事項があれば記載。なければ「特記なし」)

    ■ 患者希望
    - CT、MRIでの精査： (回答内容)
    - 骨粗鬆症の検査： (希望する / しない)
    - 希望の先生： (医師名 / 特になし)
    ---
    """
    
    gemini_history = []
    for msg in chat_history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name, system_instruction=DYNAMIC_SYSTEM_PROMPT)
            if len(gemini_history) > 1:
                chat = model.start_chat(history=gemini_history[:-1])
                response = chat.send_message(gemini_history[-1]["parts"][0])
            else:
                chat = model.start_chat(history=[])
                response = chat.send_message(gemini_history[-1]["parts"][0])
            return response.text
        except Exception:
            time.sleep(1)
            continue
    
    raise Exception("混雑のため応答できませんでした。")

# ==========================================
#  関数定義：音声認識
# ==========================================
def transcribe_audio_with_fallback(audio_file_path):
    models_to_try = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
    audio_file = genai.upload_file(path=audio_file_path)
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content(["この音声を日本語で文字起こししてください。", audio_file])
            return res.text.strip()
        except:
             continue
    return ""

# ==========================================
#  関数定義：QRコード生成
# ==========================================
def generate_qr_image(text):
    cleaned_text = format_text_for_qr(text)
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(cleaned_text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf)
    return buf

# ==========================================
#  メイン画面構成
# ==========================================
st.title("🏥 整形外科 AI問診")

# ------------------------------------------
#  Phase 1: 受付モード
# ------------------------------------------
if st.session_state.patient_data is None:
    st.info("【受付】患者情報を入力してください")
    tab1, tab2 = st.tabs(["📷 カメラで読取", "⌨️ 手動で入力"])
    
    # ★ ここでの initial_msg を変更しました！
    with tab1:
        img_file = st.camera_input("カード撮影")
        if img_file:
            with st.spinner("読み取り中..."):
                extracted = extract_patient_info(img_file)
                if extracted:
                    st.session_state.patient_data = extracted
                    greeting = get_time_based_greeting()
                    # 挨拶 ＋ 初診質問 をセットにする
                    initial_msg = f"{extracted['name']}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。\n\n早速ですが、当院へのご来院は初めてですか？"
                    st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                    st.success("成功！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("読み取り失敗")

    with tab2:
        with st.form("manual_input_form"):
            input_name = st.text_input("氏名")
            default_date = datetime.date(1980, 1, 1)
            input_dob = st.date_input("生年月日", value=default_date, min_value=datetime.date(1900, 1, 1))
            if st.form_submit_button("診察開始"):
                if input_name:
                    dob_str = input_dob.strftime("%Y年%m月%d日")
                    st.session_state.patient_data = {"name": input_name, "dob": dob_str}
                    greeting = get_time_based_greeting()
                    # 挨拶 ＋ 初診質問 をセットにする
                    initial_msg = f"{input_name}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。\n\n早速ですが、当院へのご来院は初めてですか？"
                    st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                    st.rerun()

# ------------------------------------------
#  Phase 2: 問診モード
# ------------------------------------------
else:
    p_name = st.session_state.patient_data['name']
    p_dob = st.session_state.patient_data['dob']
    
    st.caption(f"担当：那須乃アイ (Date: {today_str}) | 患者：{p_name} 様 ({p_dob})")
    
    if st.button("診察終了 / 次の患者へ"):
        st.session_state.patient_data = None
        st.session_state.messages = []
        st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                if "■ S (Subjective)" in msg["content"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(msg["content"])
                    with col2:
                        qr_image = generate_qr_image(msg["content"])
                        st.image(qr_image, caption="電子カルテ転送用QR", use_container_width=True)
                else:
                    st.markdown(msg["content"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

    audio_value = st.audio_input("マイクで回答", key=f"audio_{st.session_state.audio_key}")
    user_text_input = st.chat_input("テキストで回答")

    user_input = None
    if audio_value:
        with st.spinner("認識中..."):
            temp = f"temp_{int(time.time())}.wav"
            with open(temp, "wb") as f: f.write(audio_value.getvalue())
            try: user_input = transcribe_audio_with_fallback(temp)
            except: pass
            if os.path.exists(temp): os.remove(temp)
            st.session_state.audio_key += 1
    elif user_text_input:
        user_input = user_text_input

    if user_input and user_input != "":
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        try:
            with st.spinner("考案中..."):
                ai_text = generate_response_with_fallback(st.session_state.messages, p_name, p_dob)
            st.session_state.messages.append({"role": "assistant", "content": ai_text})
            st.rerun()
        except Exception as e:
            st.error(f"エラー: {e}")
            st.rerun()
        except Exception as e:
            st.error(f"エラー: {e}")

