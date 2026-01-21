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
    st.error("APIキー設定エラー: .streamlit/secrets.toml を確認してください。")
    st.stop()

# 今日の日付
today_str = datetime.date.today().strftime("%Y/%m/%d")

# ==========================================
#  セッション状態の初期化
# ==========================================
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None  # {name: "〇〇", dob: "yyyy/mm/dd"}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# ==========================================
#  関数定義：時間帯ごとの挨拶を取得
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
    
    # 時間帯挨拶を取得
    greeting = get_time_based_greeting()

    DYNAMIC_SYSTEM_PROMPT = f"""
    あなたは整形外科クリニックのAI問診担当「那須乃アイ（なすのあい）」です。
    以下の【問診フロー】に従って、患者と対話し、情報を収集してください。
    **本日の日付は {today_str} です。**

    【重要：患者情報】
    **現在、会話している患者の情報は以下の通りです。**
    * **氏名:** {patient_name}
    * **生年月日:** {patient_dob}
    
    この情報は既に取得済みです。
    **フローの「2. 患者情報」はスキップし、本人確認の挨拶から始めてください。**

    【最重要ルール：回答の検証】
    各ステップにおいて、患者の回答が質問に対する答えとして不適切（意味不明、全く関係ない話、聞き取りエラーによるノイズなど）な場合は、
    **絶対に次のステップに進まないでください。**
    その場合、「申し訳ありません、よく聞き取れませんでした。もう一度、〇〇について教えていただけますか？」と丁寧に聞き返し、
    適切な回答が得られるまで同じ質問を繰り返してください。

    【問診フロー】
    1. 挨拶：「{patient_name}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。」と挨拶する。
       ※生年月日の確認は行わないこと。
    2. (スキップ：患者情報は取得済み)
    3. (スキップ：名前確認済み)
    4. 初診確認：「当院へのご来院は初めてですか？」と聞く。
       - ※Yes/Noが判別できない場合は聞き直すこと。
    5. (再診の場合のみ)：「以前にかかったことがある方の場合は、診察を受けた医師名が分かる場合には教えてください」と聞く。初診ならスキップ。
    6. 来院理由：「本日はどういった症状でご来院されましたか？一番困っている症状から順に教えてください」と聞く。
    7. 主訴の確認：聞き取った症状を①②③...と箇条書きで提示し、「ということでよろしいですか？」と確認する。
       - ※ここで患者が否定した場合は、再度症状を聞き直すこと。
    8. 詳細聴取：確認が取れたら、症状ごとに詳しく聞く。
       - 発症起点（いつから）
       - 原因（外傷、スポーツ、仕事、交通事故など。事故なら詳細も聞く）
       - 増悪因子（どんな時に痛むか、動作、時間帯など）
       ※症状が複数ある場合は、一つずつ順番に聞くこと。
    9. まとめ確認：全ての聴取が終わったら、内容を簡潔にまとめて提示し、「この内容でよろしいですか？」と確認する。
    10. 画像検査：「CTやMRIでの詳細な検査をご希望されますか？」と聞く。
        選択肢を提示する：
        1. 積極的に検査を受けたい
        2. 医師が必要と判断すれば受けたい
        3. 今のところ検査を受けない
        ※「ご希望されても本日中に検査を受けられるとは限らない」ことを申し添える。
    11. 骨粗鬆症検査：「骨粗しょう症の検査を希望されますか？（はい/いいえ）」と聞く。
    12. 医師希望：「診察を希望される医師はいますか？」と聞く。
        当院の専門医：膝専門医、手専門医、足関節・足部専門医、腫瘍専門医
        具体的な医師名：(架空の名前でOK、後で設定)
        ※「特に希望しない」も選べることを伝える。
    13. 終了：「お疲れさまでした。タブレットを受付に返却して、待合でお待ちください」と案内して終了する。

    【最終出力フォーマットの厳格なルール】
    会話終了後、以下のSOAP形式で出力してください。Markdownのコードブロックは使用せず、テキストとして出力すること。

    ### 1. 日付記載の絶対ルール
    * 文中の「1週間前」「昨日」などの相対的な日時は、必ず本日（{today_str}）から逆算した「yyyy/mm/dd」形式に書き換えること。
    * **禁止事項:** 「1週間前」「数日前」という言葉をそのまま出力に残さないこと。

    ### 2. 出力テンプレート
    ---
    ■ S (Subjective)
    氏名：{patient_name} ({patient_dob}生)
    主訴：
    #1. (名詞または体言止めで簡潔に記載。例: 右膝の痛み)
    #2. (名詞または体言止めで簡潔に記載。例: 歩行困難)
    (※各項目の間には空行を入れず詰めること)

    現病歴：
    (発症起点、原因、経過などを記載。日付は必ずyyyy/mm/dd形式に変換済みのものを使用する)
    (※最終行は必ず以下の3つのうちいずれか1文で締めくくること。判断に迷う場合は2を選択)
    1. {today_str} 　症状が持続しているため当院を受診
    2. {today_str} 　症状が改善しないため当院を受診
    3. {today_str} 　症状が悪化してきたため当院を受診

    ■ O (Objective)
    (問診で得られた症状の補足事項があれば記載。なければ「特記なし」とする)

    ■ 患者希望
    - CT、MRIでの精査： (回答内容)
    - 骨粗鬆症の検査： (希望する / しない)
    - 希望の先生： (医師名 / 特になし)
    ---
    """
    
    last_error = None
    gemini_history = []
    for msg in chat_history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name, system_instruction=DYNAMIC_SYSTEM_PROMPT)
            
            if len(gemini_history) > 1:
                chat = model.start_chat(history=gemini_history[:-1])
                last_user_message = gemini_history[-1]["parts"][0]
                response = chat.send_message(last_user_message)
            else:
                chat = model.start_chat(history=[])
                last_user_message = gemini_history[-1]["parts"][0]
                response = chat.send_message(last_user_message)

            return response.text

        except ResourceExhausted:
            st.toast(f"⚠️ {model_name} が混雑中。{models_to_try[models_to_try.index(model_name)+1] if models_to_try.index(model_name)+1 < len(models_to_try) else '終了'} に切り替えます...")
            time.sleep(1) 
            last_error = "ResourceExhausted"
            continue
        except Exception as e:
            raise e
    
    raise Exception(f"全てのモデルが混雑していました。({last_error})")

# ==========================================
#  関数定義：音声認識
# ==========================================
def transcribe_audio_with_fallback(audio_file_path):
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash", 
        "gemini-3-flash-preview", 
        "gemini-2.0-flash"
    ]
    audio_file = genai.upload_file(path=audio_file_path)
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content(["この音声を日本語で文字起こししてください。", audio_file])
            try:
                return res.text.strip()
            except ValueError:
                return "" 
        except:
             time.sleep(1)
             continue
    return ""

def generate_qr_image(text):
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data(text)
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
#  Phase 1: 受付モード (データがない場合)
# ------------------------------------------
if st.session_state.patient_data is None:
    st.info("【受付】患者情報を入力してください")
    
    # ★変更点: タブで入力方法を切り替え
    tab1, tab2 = st.tabs(["📷 カメラで読取", "⌨️ 手動で入力"])
    
    # --- タブ1: カメラ入力 ---
    with tab1:
        img_file = st.camera_input("カード撮影")
        if img_file:
            with st.spinner("情報を読み取っています..."):
                extracted = extract_patient_info(img_file)
                if extracted:
                    st.session_state.patient_data = extracted
                    # 時間帯挨拶
                    greeting = get_time_based_greeting()
                    initial_msg = f"{extracted['name']}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。"
                    st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                    st.success("読み取り成功！問診を開始します。")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("読み取れませんでした。もう一度撮影するか、手動入力を試してください。")

    # --- タブ2: 手動入力 ---
    with tab2:
        with st.form("manual_input_form"):
            input_name = st.text_input("氏名 (例: 山田 太郎)")
            # 生年月日入力（デフォルトは昭和50年あたりにしておく）
            default_date = datetime.date(1980, 1, 1)
            input_dob = st.date_input("生年月日", value=default_date, min_value=datetime.date(1900, 1, 1))
            
            submitted = st.form_submit_button("診察開始")
            
            if submitted:
                if input_name:
                    # 日付を文字列に変換 (yyyy年mm月dd日)
                    dob_str = input_dob.strftime("%Y年%m月%d日")
                    st.session_state.patient_data = {
                        "name": input_name,
                        "dob": dob_str
                    }
                    # 時間帯挨拶
                    greeting = get_time_based_greeting()
                    initial_msg = f"{input_name}さん、{greeting}。那須乃あいです。本日の問診を担当させていただきます。よろしくお願い致します。"
                    st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                    st.rerun()
                else:
                    st.warning("氏名を入力してください。")

# ------------------------------------------
#  Phase 2: 問診モード (データがある場合)
# ------------------------------------------
else:
    p_name = st.session_state.patient_data['name']
    p_dob = st.session_state.patient_data['dob']
    
    st.caption(f"担当：那須乃アイ (Date: {today_str}) | 患者：{p_name} 様 ({p_dob})")
    
    # データをリセットするボタン（次の患者さんへ）
    if st.button("診察終了 / 次の患者へ"):
        st.session_state.patient_data = None
        st.session_state.messages = []
        st.rerun()

    # --- チャット履歴表示 ---
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

    # --- 入力エリア ---
    audio_value = st.audio_input("マイクで回答する", key=f"audio_{st.session_state.audio_key}")
    user_text_input = st.chat_input("テキストで回答する")

    user_input = None

    if audio_value:
        with st.spinner("音声を認識中..."):
            temp_filename = f"temp_{int(time.time())}.wav"
            with open(temp_filename, "wb") as f:
                f.write(audio_value.getvalue())
            try:
                user_input = transcribe_audio_with_fallback(temp_filename)
            except:
                pass
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            st.session_state.audio_key += 1

    elif user_text_input:
        user_input = user_text_input

    # --- 会話進行 ---
    if user_input and user_input != "":
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        try:
            with st.spinner("那須乃アイさんが考えています..."):
                ai_response_text = generate_response_with_fallback(st.session_state.messages, p_name, p_dob)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
            st.rerun()

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
