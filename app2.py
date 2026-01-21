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

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("APIキー設定エラー: .streamlit/secrets.toml (またはCloudのSecrets) を確認してください。")
    st.stop()

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
if "interview_state" not in st.session_state:
    st.session_state.interview_state = "chat" # chat, form, complete

# ==========================================
#  関数定義群
# ==========================================
def get_time_based_greeting():
    hour = datetime.datetime.now().hour
    if hour < 10: return "おはようございます"
    elif hour < 18: return "こんにちは"
    else: return "こんばんは"

def format_text_for_qr(text):
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if "氏名：" in line and "生)" in line: continue
        clean_line = line.replace("■ S (Subjective)", "【S】")
        clean_line = clean_line.replace("■ O (Objective)", "【O】")
        clean_line = clean_line.replace("■ 患者希望", "【P】")
        clean_line = clean_line.replace("■ Plan", "【P】")
        if "---" in clean_line: continue
        new_lines.append(clean_line)
    formatted_text = "\r\n".join(new_lines)
    while "\r\n\r\n\r\n" in formatted_text:
        formatted_text = formatted_text.replace("\r\n\r\n\r\n", "\r\n\r\n")
    return formatted_text.strip()

def extract_patient_info(image_data):
    model = genai.GenerativeModel('gemini-2.5-flash')
    img = Image.open(image_data)
    prompt = """
    この身分証（マイナンバーカード等）の画像から、以下の情報を読み取ってください。
    1. 氏名（漢字）
    2. 生年月日（西暦yyyy年mm月dd日形式に変換）
    出力はJSON形式のみで行ってください。
    ```json
    { "name": "氏名", "dob": "yyyy年mm月dd日" }
    ```
    """
    try:
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        json_str = text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        return None

def transcribe_audio_with_fallback(audio_file_path):
    models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
    audio_file = genai.upload_file(path=audio_file_path)
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            res = model.generate_content(["この音声を日本語で文字起こししてください。", audio_file])
            return res.text.strip()
        except: continue
    return ""

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
#  AI応答生成（チャット用）
# ==========================================
def generate_chat_response(chat_history, patient_name, patient_dob):
    models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-flash-preview"]
    greeting = get_time_based_greeting()

    SYSTEM_PROMPT = f"""
    あなたは整形外科クリニックのAI問診担当「那須乃アイ」です。
    本日は {today_str} です。
    患者：{patient_name} ({patient_dob}) ※取得済み

    【タスク】
    患者の主訴、発症時期、原因、経過（現病歴）を聴取してください。
    
    【重要ルール】
    1. 挨拶＆初診確認：「{patient_name}さん、{greeting}。那須乃あいです...当院へのご来院は初めてですか？」から開始。
    2. 症状の聴取が十分に終わったら、**これ以上質問せず**、以下の終了合図のみを出力してください。
       出力： <END_OF_INTERVIEW>
    3. **まだSOAPまとめは出力しないでください。**
    4. 画像検査、骨粗鬆症検査、医師希望については**質問しないでください**（後で画面で入力します）。

    【会話スタイル】
    丁寧かつ簡潔に。1回の発言で質問は1つまで。
    """
    
    gemini_history = [{"role": "model" if m["role"]=="assistant" else "user", "parts": [m["content"]]} for m in chat_history]

    for m_name in models:
        try:
            model = genai.GenerativeModel(m_name, system_instruction=SYSTEM_PROMPT)
            chat = model.start_chat(history=gemini_history[:-1] if len(gemini_history)>1 else [])
            response = chat.send_message(gemini_history[-1]["parts"][0])
            return response.text
        except: continue
    raise Exception("応答できませんでした")

# ==========================================
#  最終SOAP作成
# ==========================================
def generate_final_soap(chat_history, patient_name, patient_dob, selection_data):
    models = ["gemini-3-flash-preview", "gemini-2.5-flash"]
    
    plan_text = f"""
    - 画像検査希望: {selection_data['image_exam']}
    - 骨粗鬆症検査: {selection_data['osteo_exam']}
    - 医師希望: {selection_data['doctor']}
    """

    PROMPT = f"""
    これまでの会話履歴をもとに、整形外科の電子カルテ用SOAP（S部分）を作成し、
    最後に以下の患者希望情報（P部分）を結合して出力してください。

    【患者希望情報】
    {plan_text}

    【出力フォーマットの厳格なルール】
    Markdownコードブロックは使用しないこと。

    ### 1. 日付記載の絶対ルール (基準日: {today_str})
    * 相対日付（昨日、2週間前など）は必ず {today_str} から逆算した「yyyy/mm/dd」形式に変換。
    * 「1週間前」「昨日」などの言葉は禁止。
    * 1ヶ月前→「yyyy/mm月上旬/中旬/下旬」
    * 半年以上前→「yyyy/mm月下旬頃」または年単位

    ### 2. 現病歴のレイアウト（主訴ごとにグループ化）
    #1. (主訴)
    yyyy/mm/dd (経過記述...)
    yyyy/mm/dd (受診理由...)
    
    (空行)

    #2. (主訴)
    yyyy/mm/dd (経過記述...)
    ...

    ### 出力テンプレート
    ---
    ■ S (Subjective)
    氏名：{patient_name} ({patient_dob}生)
    主訴：
    #1. (体言止め)
    #2. (体言止め)

    現病歴：
    (上記レイアウトに従って記述)
    (※各主訴ブロックの最終行は必ず以下のいずれかで締める)
    1. {today_str} 　症状が持続しているため当院を受診
    2. {today_str} 　症状が改善しないため当院を受診
    3. {today_str} 　症状が悪化してきたため当院を受診

    ■ O (Objective)
    (会話から分かる特記あれば記載、なければ「特記なし」)

    ■ 患者希望
    {plan_text}
    ---
    """
    
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    
    for m_name in models:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content([PROMPT, conversation_text])
            return response.text
        except: continue
    return "エラー：カルテ作成に失敗しました"

# ==========================================
#  メイン画面構成
# ==========================================
st.title("🏥 整形外科 AI問診")

# --- 1. 受付フェーズ ---
if st.session_state.patient_data is None:
    st.info("【受付】患者情報を入力してください")
    tab1, tab2 = st.tabs(["📷 カメラで読取", "⌨️ 手動で入力"])
    
    with tab1:
        img_file = st.camera_input("カード撮影")
        if img_file:
            with st.spinner("読み取り中..."):
                extracted = extract_patient_info(img_file)
                if extracted:
                    st.session_state.patient_data = extracted
                    st.session_state.messages = []
                    greeting = get_time_based_greeting()
                    initial_msg = f"{extracted['name']}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。\n\n早速ですが、当院へのご来院は初めてですか？"
                    st.session_state.messages.append({"role": "assistant", "content": initial_msg})
                    st.rerun()

    with tab2:
        with st.form("manual"):
            name = st.text_input("氏名")
            dob = st.date_input("生年月日", value=datetime.date(1980,1,1), min_value=datetime.date(1900,1,1))
            if st.form_submit_button("診察開始"):
                dob_str = dob.strftime("%Y年%m月%d日")
                st.session_state.patient_data = {"name": name, "dob": dob_str}
                st.session_state.messages = []
                greeting = get_time_based_greeting()
                initial_msg = f"{name}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。\n\n早速ですが、当院へのご来院は初めてですか？"
                st.session_state.messages.append({"role": "assistant", "content": initial_msg})
                st.rerun()

# --- 2. 問診＆フォームフェーズ ---
else:
    p_name = st.session_state.patient_data['name']
    p_dob = st.session_state.patient_data['dob']
    
    st.caption(f"担当：那須乃アイ | 患者：{p_name} 様 ({p_dob})")
    
    if st.button("診察終了 / 次の患者へ"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                if "<END_OF_INTERVIEW>" in msg["content"]:
                    st.write("（問診終了。下のフォームに入力してください）")
                elif "■ S (Subjective)" in msg["content"]:
                     c1, c2 = st.columns([3, 1])
                     with c1: st.markdown(msg["content"])
                     with c2: st.image(generate_qr_image(msg["content"]), caption="カルテ転送用QR")
                else:
                    st.markdown(msg["content"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

    is_chatting = st.session_state.interview_state == "chat"

    if is_chatting:
        audio_val = st.audio_input("マイク", key=f"aud_{st.session_state.audio_key}")
        text_val = st.chat_input("回答を入力")
        user_input = None

        if audio_val:
            with st.spinner("認識中..."):
                tmp = f"tmp_{int(time.time())}.wav"
                with open(tmp, "wb") as f: f.write(audio_val.getvalue())
                try: user_input = transcribe_audio_with_fallback(tmp)
                except: pass
                if os.path.exists(tmp): os.remove(tmp)
                st.session_state.audio_key += 1
        elif text_val:
            user_input = text_val

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("思考中..."):
                ai_res = generate_chat_response(st.session_state.messages, p_name, p_dob)
                if "<END_OF_INTERVIEW>" in ai_res:
                    st.session_state.interview_state = "form"
                    st.session_state.messages.append({"role": "assistant", "content": "<END_OF_INTERVIEW>"})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": ai_res})
                st.rerun()

    else:
        # --- 3. フォーム入力フェーズ (ここを大幅改良！) ---
        if st.session_state.interview_state == "form":
            st.divider()
            st.subheader("📋 最終確認")
            
            with st.form("final_options"):
                st.markdown("以下の項目を選択して、カルテを作成してください。")
                
                # --- 画像検査 ---
                st.markdown("#### 画像検査")
                img_opt = st.radio(
                    "CTやMRIでの詳しい検査を希望されますか？", 
                    ["積極的に検査を受けたい", "医師が必要と判断すれば検査を受けたい", "くわしい検査はいまのところ希望しない"]
                )
                st.caption("※当日の予約状況により、本日中に検査が受けられない場合もございます。あらかじめご了承ください。")
                st.divider()

                # --- 骨粗鬆症検査 ---
                st.markdown("#### 骨粗鬆症検査")
                st.info("💡 60代以降の女性の方は、一度骨粗鬆症の検査を行うことをおすすめします。")
                osteo_opt = st.radio(
                    "骨粗鬆症の検査をご希望されますか？", 
                    ["はい", "いいえ"],
                    horizontal=True
                )
                st.divider()
                
                # --- 医師希望 ---
                st.markdown("#### 医師希望")
                doc_cat = st.radio(
                    "本日の診察を担当する医師にご希望の医師はございますか？",
                    ["手の専門医", "膝の専門医", "足関節、足部（膝から下）の専門医", "特に希望はない", "医師名を指定する"]
                )
                
                # 名前指定の場合の入力欄
                doc_name_input = st.text_input("医師名（※上記で「医師名を指定する」を選択した場合のみ記入）")
                
                st.divider()
                
                if st.form_submit_button("✅ カルテ作成"):
                    # 医師名のロジック処理
                    if doc_cat == "医師名を指定する" and doc_name_input:
                        final_doc = f"指定あり: {doc_name_input}"
                    elif doc_cat == "医師名を指定する":
                        final_doc = "指定あり (名前未記入)"
                    else:
                        final_doc = doc_cat

                    selections = {
                        "image_exam": img_opt,
                        "osteo_exam": osteo_opt,
                        "doctor": final_doc
                    }
                    
                    with st.spinner("SOAPを作成中..."):
                        final_soap = generate_final_soap(st.session_state.messages, p_name, p_dob, selections)
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_soap})
                    st.session_state.interview_state = "complete" 
                    st.rerun()

