"""

②. DBpediaから固有名詞をリストアップするコード
①で選んだ中カテゴリ(Mid_classes)に属する固有名詞を、DBpediaのSPARQLエンドポイントから大量に取ってくる

元となったコード：
* notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb 内の⭐️部分
    * queryの試行錯誤はこのノートブック内で行った

uv run python src/listup_properNouns.py
"""
import os
import sys
import json
import time
import random
import re
import csv
import requests
from tqdm import tqdm

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(project_root))


ENDPOINT = "https://query.wikidata.org/sparql"


# ===== 設定 =====
LIMIT = 1000 #2000                                      # まずは 2000〜10000 で調整
SLEEP_SEC = 0.0                                         # リクエスト間隔
MAX_RETRIES = 8                                         # リトライ回数
TIMEOUT_SEC = 60

STATE_FILE = os.path.join(project_root, "data", "dbpedia", f"wikidata_state_{LIMIT}.txt")    # 途中再開用に最後のOFFSETを記録 # 旧: os.path.join(project_root, "data", f"wikidata_buildings_state_{LIMIT}.txt")
OUT_CSV_DIR = os.path.join(project_root, "data", "dbpedia", f"wikidata_Things_childs_LIMIT{LIMIT}") # 旧: "data", f"wikidata_Things_childs_LIMIT{LIMIT}")
os.makedirs(OUT_CSV_DIR, exist_ok=True)

mid_classes_path = os.path.join(project_root, "data", "dbpedia", "Mid_classes.json") # 旧: os.path.join(project_root, "data", "dbpedia_selected_subclasses.json")
QID_map_path = os.path.join(project_root, "data", "dbpedia", "Mid_class_QID_map.json") # 旧: os.path.join(project_root, "data", "dbpedia_to_wikidata_qid_map_20260117.json")




DBPEDIA_SPARQL = "https://dbpedia.org/sparql"


query_base = """PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?wikidata WHERE {
  <CLASS_URL>
    owl:equivalentClass ?wikidata .
}
"""


def run_sparql(query: str, timeout: int = 30) -> dict:
    headers = {
        # できれば連絡先つきのUAにする（エンドポイントに優しい）
        "User-Agent": "my-research-bot/0.1 (contact: you@example.com)",
        "Accept": "application/sparql-results+json",
    }
    params = {
        "query": query,
        "format": "json",  # DBpediaはこれでOK
    }
    r = requests.get(DBPEDIA_SPARQL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

# =====

QID_RE = re.compile(r"Q\d+")
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "MyWikidataDownloader/1.0" # (your_email@example.com)"
}

# 条件: (このくらいまで絞る+filterのタイミングを調整してなるべく効率的に候補を絞らないと、すぐにtimeoutエラーになる)
# * まず, 英語Wikipedia記事があるものだけ（強力に候補を減らす）(=有名な建物に絞る)
# * building直下のみに限定
# * 座標があるものに限定（=実在する建物に絞る）
# * ノイズ除去: 曖昧さ回避とリスト記事
# * sitelinks 順に並べ、上位LIMIT個を取得 (=有名な建物に絞る)

SPARQL_1HOP_MostSiteLinked_base_for_building = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX schema: <http://schema.org/>

SELECT ?item ?itemLabel ?type ?typeLabel ?sitelinks ?coord WHERE {

  {
    SELECT ?item (SAMPLE(?type0) AS ?type) (SAMPLE(?coord0) AS ?coord) (MAX(?sitelinks0) AS ?sitelinks) WHERE {

        # 英語Wikipedia記事があるものだけ（強力に候補を減らす）
        ?article schema:about ?item ;
                schema:isPartOf <https://en.wikipedia.org/> .

        # building（またはそのサブクラス）に属する
        ?item wdt:P31 ?type0 .
        ?type0 wdt:P279 wd:<<QID>> . # ?type0 wdt:P279* wd:Q41176 .だと重すぎて動かなかった。ので*を消して1-hopに限定
        # 次のように2-hopもダメだった
        # { ?type wdt:P279 wd:<<QID>> . }
        # UNION
        # { ?type wdt:P279/wdt:P279 wd:<<QID>> . }


        # 座標あり
        ?item wdt:P625 ?coord0 .

        # sitelinks
        ?item wikibase:sitelinks ?sitelinks0 .

        # ノイズ除去（任意）
        FILTER NOT EXISTS { ?item wdt:P31 wd:Q4167410 . }   # disambiguation
        FILTER NOT EXISTS { ?item wdt:P31 wd:Q13406463 . }  # list article
    }
    GROUP BY ?item
    ORDER BY DESC(?sitelinks)
    LIMIT <<LIMIT_NUM>> # 👉 {} の中（サブクエリ）の LIMIT を小さくすると、sitelinks が多い順に並べた結果の「上位の一部だけ」が候補として抽出され
  }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?sitelinks)
"""


SPARQL_1HOP_MostSiteLinked_base = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX schema: <http://schema.org/>

SELECT ?item ?itemLabel ?type ?typeLabel ?sitelinks WHERE {

  {
    SELECT ?item (SAMPLE(?type0) AS ?type) (MAX(?sitelinks0) AS ?sitelinks) WHERE {

        # 英語Wikipedia記事があるものだけ（強力に候補を減らす）
        ?article schema:about ?item ;
                schema:isPartOf <https://en.wikipedia.org/> .

        # building（またはそのサブクラス）に属する
        ?item wdt:P31 ?type0 .
        ?type0 wdt:P279 wd:<<QID>> . # ?type0 wdt:P279* wd:Q41176 .だと重すぎて動かなかった。ので*を消して1-hopに限定
        # 次のように2-hopもダメだった
        # { ?type wdt:P279 wd:<<QID>> . }
        # UNION
        # { ?type wdt:P279/wdt:P279 wd:<<QID>> . }

        # sitelinks
        ?item wikibase:sitelinks ?sitelinks0 .

        # ノイズ除去（任意）
        FILTER NOT EXISTS { ?item wdt:P31 wd:Q4167410 . }   # disambiguation
        FILTER NOT EXISTS { ?item wdt:P31 wd:Q13406463 . }  # list article
    }
    GROUP BY ?item
    ORDER BY DESC(?sitelinks)
    LIMIT <<LIMIT_NUM>> # 👉 {} の中（サブクエリ）の LIMIT を小さくすると、sitelinks が多い順に並べた結果の「上位の一部だけ」が候補として抽出され
  }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?sitelinks)
"""

'''
[memo] 上で使っているP279の意味について:
Wikidataでは
・QID = エンティティ
・P31 = instance of（〜の一種）
・P279 = subclass of（〜の下位概念）
を使って、
「建築物（building / architectural structure）」
またはその下位クラスに属するもの
をすべて引っ張ります。
'''


def load_offset(default=0) -> int:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = f.read().strip()
            if s.isdigit():
                return int(s)
    return default

def save_offset(offset: int) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.write(str(offset))

def build_query(SPARQL_1HOP_MostSiteLinked, limit: int, offset: int) -> str:
    SPARQL_1HOP_MostSiteLinked = SPARQL_1HOP_MostSiteLinked.strip() + f"\nOFFSET {offset}\n"
    return SPARQL_1HOP_MostSiteLinked.replace("<<LIMIT_NUM>>", str(limit))


def fetch_page(SPARQL_1HOP_MostSiteLinked, limit: int, offset: int, class_label: str, class_QID: str):
    query = build_query(SPARQL_1HOP_MostSiteLinked, limit, offset)
    params = {"query": query}

    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        r = None
        try:
            r = requests.get(ENDPOINT, params=params, headers=HEADERS, timeout=TIMEOUT_SEC)

            # まずHTTPコードで弾く（既存 + 追加）
            if r.status_code in (429, 503, 502, 504, 500, 520, 522, 524, 403):
                wait = backoff + random.uniform(0, 0.5)
                print(f"[WARN] HTTP {r.status_code} at offset={offset}. retry in {wait:.1f}s (attempt {attempt})")
                # デバッグ：本文先頭を少しだけ
                print(f"       content-type={r.headers.get('content-type')} body[:200]={r.text[:200]!r}")
                time.sleep(wait)
                backoff *= 2
                continue

            r.raise_for_status()

            # JSONでない場合（HTML等）を検出してリトライ
            ctype = (r.headers.get("content-type") or "").lower()
            if "json" not in ctype:
                wait = backoff + random.uniform(0, 0.5)
                print(f"[WARN] Non-JSON response at offset={offset} ctype={ctype}. retry in {wait:.1f}s (attempt {attempt})")
                print(f"       body[:200]={r.text[:200]!r}")
                time.sleep(wait)
                backoff *= 2
                continue

            # ここで初めてJSONパース（失敗したらリトライ）
            try:
                data = r.json()
            except ValueError:
                wait = backoff + random.uniform(0, 0.5)
                print(f"[WARN] JSON decode failed at offset={offset}. retry in {wait:.1f}s (attempt {attempt})")
                print(f"       body[:200]={r.text[:200]!r}")
                time.sleep(wait)
                backoff *= 2
                continue

            bindings = data["results"]["bindings"]
            rows = []
            for b in bindings:
                item_url = b["item"]["value"]
                m = QID_RE.search(item_url)
                qid = m.group(0) if m else item_url
                label = b.get("itemLabel", {}).get("value", "")
                # rows.append((qid, label))
                rows.append((qid, label, class_label, class_QID))
            return rows

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = backoff + random.uniform(0, 0.5)
            print(f"[WARN] {type(e).__name__} at offset={offset}. retry in {wait:.1f}s (attempt {attempt})")
            time.sleep(wait)
            backoff *= 2
        except requests.HTTPError as e:
            body = ""
            if r is not None:
                body = (r.text or "")[:200]
            print(f"[ERROR] HTTPError at offset={offset}: {e} body[:200]={body!r}")
            raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries at offset={offset}")


def append_rows_to_csv(rows, csv_path: str):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            # w.writerow(["qid", "label"])
            w.writerow(["qid", "label", "class_label", "class_qid"]) # [memo] 全てのsubclassを1つのCSVにまとめる場合はどのクラス由来か分かるようにこれを使う. rowにclass_label, class_qidも追加する必要あり
        w.writerows(rows)




# =================================================================================

def main():
    # 保存した，選択したサブクラスを読み込む
    with open(mid_classes_path, "r") as f:
        mid_classes = json.load(f)


    # ****** クラスのQIDを取得し保存する ******
    QID_map = {}
    already_read = set()
    # already_read.update(QID_map.keys())
    for class_url, name in mid_classes:
        if name in already_read:
            continue
        already_read.add(name)
        query = query_base.replace("<CLASS_URL>", f"<{class_url}>")
        data = run_sparql(query)
        wikidata_ids = [b["wikidata"]["value"] for b in data["results"]["bindings"]]
        print(f"- {name} -> {class_url} -> {wikidata_ids}")
        QIDs = []
        for wikidata_id in wikidata_ids:
            QID = wikidata_id.split("/")[-1] if wikidata_id else None # e.g. ''http://www.wikidata.org/entity/Q39614' -> 'Q39614'
            # time.sleep(0.2)  # 公開endpointに優しく
            if QID.startswith("Q"):
                # 未登録ならlistを，登録済ならそのlistに追加
                # QID_map.setdefault(name, []).append(QID)
                QIDs.append(QID)
                QID_map[name] = list(set(QIDs)) # 重複削除
            
    # 保存
    with open(QID_map_path, "w") as f:
        json.dump(QID_map, f, indent=2)

    

    # ****** 各クラス（カテゴリ）から， 直下の固有名詞を取得する ******
    # * サブクラス毎に，固有名詞をCSV保存する

    # QID_map読み込み
    with open(QID_map_path, "r") as f:
        QID_map = json.load(f)

    # 今回はoffsetは使わない．SPARQL_1HOP_MostSiteLinked毎にoffsetが変わるためいちいち保存したくないため．
    # result = {}
    for class_url, class_label in tqdm(mid_classes, desc="Processing mid-classes"):    
        # class_labelが英語以外ならスキップ
        if not re.match(r'^[a-zA-Z0-9\s]+$', class_label):
            print(f"Skipping non-English class label: {class_label}")
            continue
        QIDs = QID_map.get(class_label, [None])
        OUT_CSV = os.path.join(OUT_CSV_DIR, f"{class_label.replace(' ', '_')}.csv")
        
        # 既にCSVがある場合、重複排除用に既存QIDを読み込む（巨大なら別方式推奨）
        seen = set()
        if os.path.exists(OUT_CSV):
            with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    seen.add(row["qid"])
            print(f"Loaded {len(seen)} existing qids from CSV for de-dup")

        for QID in QIDs:
            if QID is None:
                # print(f"- {class_label} -> No QID found, skipping")
                continue
            if not QID.startswith("Q"):
                continue
            if QID == 'Q9259':
                continue
            print(f"- {class_label} -> {QID}")
            
            if class_label.lower().find("building") >= 0:
                SPARQL_1HOP_MostSiteLinked = SPARQL_1HOP_MostSiteLinked_base_for_building
            else:
                SPARQL_1HOP_MostSiteLinked = SPARQL_1HOP_MostSiteLinked_base
            SPARQL_1HOP_MostSiteLinked = SPARQL_1HOP_MostSiteLinked.replace("<<LIMIT_NUM>>", str(LIMIT))
            SPARQL_1HOP_MostSiteLinked = SPARQL_1HOP_MostSiteLinked.replace("<<QID>>", QID)

            # 既にCSVがある場合、重複排除用に既存QIDを記録（巨大なら別方式推奨）
            seen = set()
            if os.path.exists(OUT_CSV):
                with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        seen.add(row["qid"])
                print(f"Loaded {len(seen)} existing qids from CSV for de-dup")

            offset=0
            pages_fetched = 0
            while True:
                if pages_fetched >= 5:
                    # 取得するページ数を制限する
                    print("Fetched 5 pages, moving to next class.")
                    break

                rows = fetch_page(SPARQL_1HOP_MostSiteLinked, LIMIT, offset, class_label, QID)
                if not rows:
                    print("No more rows. Done.")
                    break
                pages_fetched += 1

                # 重複排除（QID基準）
                # new_rows = [(qid, label) for (qid, label) in rows if qid not in seen]
                new_rows = [(qid, label, class_label, class_QID) for (qid, label, class_label, class_QID) in rows if qid not in seen]
                for qid, _, _, _ in new_rows:
                    seen.add(qid)

                # CSVに追記保存
                append_rows_to_csv(new_rows, OUT_CSV)
                print(f"offset={offset} fetched={len(rows)} appended={len(new_rows)} total_seen={len(seen)}")

                offset += LIMIT
                save_offset(offset)

                time.sleep(SLEEP_SEC)
    
    return


if __name__ == "__main__":
    main()







        
