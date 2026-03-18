


# ========================= 生成用関数 =========================

def gen_with_google_genai_api(client, prompt, schema, temperature=0.2, topP=0.8, model="gemini-2.5-flash-lite", max_retries=1):
    """Google GenAI APIを呼び出して、指定したスキーマに沿った応答を生成する。
    Args:
        client (google.generativeai.GenerativeModel): Google GenAI APIのクライアント。
        prompt (str): プロンプトのテキスト。
        schema (dict): 生成された応答が従うべきJSONスキーマ。
        temperature (float): 生成時の温度パラメータ。デフォルトは0.2。
        topP (float): 生成時のtopPパラメータ。デフォルトは0.8。
        model (str): 使用するモデル名。デフォルトは"gemini-2.5-flash-lite"。
        max_retries (int): API呼び出しの最大リトライ回数。デフォルトは1。
    Returns:
        response をそのまま返す
        ×dict: 生成された応答がスキーマに従っている場合はその応答を辞書形式で返す。スキーマに従っていない場合はNoneを返す。
    """
    schema["temperature"] = temperature  # スキーマに温度パラメータを追加
    schema["topP"] = topP  # スキーマにtopPパラメータを追加

    # *** Gemini で生成 ***
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=schema
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                raise e
            else:
                continue
        # APIからの応答を取得
        return response
    return None




def gen_with_openai_api(client, prompt, schema, temperature, topP, model="gpt-5-mini", max_retries=1):
    """OpenAI APIを呼び出して、指定したスキーマに沿った応答を生成する。
    Args:
        client (openai.OpenAI): OpenAI APIのクライアント。
        prompt (str): プロンプトのテキスト。
        schema (dict): 生成された応答が従うべきJSONスキーマ。
        temperature (float): 生成時の温度パラメータ。デフォルトは0.2。
        topP (float): 生成時のtopPパラメータ。デフォルトは0.8。
        model (str): 使用するモデル名。デフォルトは"gpt-5-mini"。
        max_retries (int): API呼び出しの最大リトライ回数。デフォルトは1。
    Returns:
        response をそのまま返す
        ×dict: 生成された応答がスキーマに従っている場合はその応答を辞書形式で返す。スキーマに従っていない場合はNoneを返す。
    """

    # *** GPT5で生成 ***
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                text={
                    "format": schema
                },
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                raise e
            else:
                continue
        # APIからの応答を取得
        return response
    return None