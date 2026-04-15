
import os
import sys
import json
import re
import wikipediaapi

project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
from utils.handle_text_utils import change_propnoun_to_filename

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
        exists = page.exists()
    except KeyError:
        print("APIレスポンス異常")
        exists = False

    # if not page.exists():
    if not exists:
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




def load_wikisummary(propnoun, wiki_pages_dir):
    """dbpediaから収集した固有名詞のwikipedia summaryを読み込んで、summaryを返す。
    data/wiki_pages に未保存であれば、data dir もしくは wiki apiから取得して、summaryを返す。
    """
    # print("Loading Wikipedia summaries for prop nouns...")
    # wiki_pages_dir = os.path.join(project_root, "data", "wiki_pages")

    filename = change_propnoun_to_filename(propnoun) + ".json"  # ファイル名に使用できない文字を置換
    wikipage_path = os.path.join(wiki_pages_dir, filename)
    
    # * 未取得の場合、wikipedia apiから取得して保存する
    if not os.path.exists(wikipage_path):
        wiki_info = fetch_wikipedia_page(propnoun, lang="en")
        if wiki_info["exists"] == False:
            print(f"Wikipedia page for concept '{propnoun}' DOES NOT exist. Skipping generation.")
            return None
        # 本文を切り出す
        main_text = extract_wiki_main_text(wiki_info['text'])
        wiki_info['text'] = main_text

        # 保存
        with open(wikipage_path, "w") as f:
            json.dump(wiki_info, f, ensure_ascii=False, indent=4)

    # * 今ここで保存した or すでに保存されているwikipedia summaryを読み込む
    with open(wikipage_path, "r") as f:
        wiki_page = json.load(f)
        summary = wiki_page.get("summary")
        if summary:
            # self.propnoun_to_wikisummary[propnoun] = summary
            # print(f"Loaded Wikipedia summary for '{propnoun}' from wiki_pages.")
            return summary
        else:
            print(f"No summary found in wiki page for '{propnoun}' in wiki_pages.")
            return None
        


