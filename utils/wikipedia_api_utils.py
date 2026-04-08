

import re
import wikipediaapi


# ========================= wikipedia page 取得用関数 =========================
def fetch_wikipedia_page(title: str, lang: str = "en") -> dict:
    """Fetch a Wikipedia page by title and language.
    wiki pageを取得する．
    Args:
        title (str): Wikipediaページのタイトル
        lang (str): Wikipediaの言語コード（例: "en", "ja"）

    Returns:
        dict: ページ情報を含む辞書．ページが存在しない場合は'exists'キーがFalseとなる．

    Note:
        * titleに完全一致するページのみが取得されるため，google検索のように最も近いが異なるものを検索結果として返すことはない．
        * そのため，提示した固有名詞とは違う情報のページが返されることは無い．
        * また，titleの綴りが間違っていても, userがよく間違う綴りの場合は, wikipedia側で正しいページにリダイレクトしてくれる．
            * 例: "Colloseum" -> "Colosseum"のページを取得
    
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="my-wikipedia-fetcher/1.0 (contact: your_email@example.com)",
        language=lang,
    )
    page = wiki.page(title)

    try:
        if not page.exists():
            return {"exists": False, "title": title, "lang": lang}
    except Exception as e:
        print(f"Error occurred while fetching Wikipedia page: {e}")
        return {"exists": False, "title": title, "lang": lang}

    return {
        "exists": True,
        "title": page.title,
        "url": page.fullurl,
        "summary": page.summary,
        "text": page.text,  # 全文（長いので注意）
    }


def extract_wiki_main_text(text: str) -> str:
    """Wikipediaページの情報から本文テキストのみを抽出する。
    Args:
        text (str): Wikipediaページの全文

    Returns:
        str: Wikipediaページの本文テキスト。ページが存在しない場合は空文字列を返す。
    """

    STOP_SECTIONS = {
        "see also",
        "references",
        "notes",
        "further reading",
        "external links",
        "bibliography",
        "sources",
        "works cited",
        "citations",
    }
    # 見出し候補を | で連結
    titles = "|".join(re.escape(x) for x in STOP_SECTIONS)

    # 行頭に単独で出る見出しを検出
    # 例:
    # See also
    # References
    #
    # または wiki風:
    # == See also ==
    pattern = rf"(?mi)^\s*(?:=+\s*)?(?:{titles})(?:\s*=+)?\s*$"

    # 最初に見つかった見出し以降を全て削除
    m = re.search(pattern, text)
    if m:
        text = text[:m.start()]

    # 脚注番号 [1], [23] を削除
    text = re.sub(r"\[\d+\]", "", text)

    # 空行を整理
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
