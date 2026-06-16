"""
script概要:
DBpediaのカテゴリのうち、本研究に使えそうなカテゴリのみを選別するためのスクリプト。
- 元は人手で確認し大カテゴリlistをfilteringしていた、`notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb` 内の「実体を持つクラスを選ぶ」に当たる。
- このファイルを整理し、src_data/filter_DBpedia_Top_categories.py を作成した。

修正完了日: 2026/06/07

注意点:
実行時間が長い listup_proper_nouns_from_subclasses はコメントアウトしている。
- listup_proper_nouns_from_subclasses は以前のrepositoryで実施済みの処理であるため、再度の実行によるコードの正確性の検証はしていないことに注意。

実行コマンド:
```
uv run python src_data/filter_DBpedia_Top_categories.py
```
"""


import os
import sys
from typing import Dict, List, Tuple

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.dbpedia_api_utils import run_sparql, load_offset, save_offset, build_query, fetch_page

import json
import time
import random
import re
import csv
import requests
import pandas as pd

ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"


# ===== 設定 =====
LIMIT = 1000 #2000                                      # まずは 2000〜10000 で調整
SLEEP_SEC = 1.0                                         # リクエスト間隔
MAX_RETRIES = 8                                         # リトライ回数
TIMEOUT_SEC = 60
OUT_CSV = os.path.join(project_root, "data", f"wikidata_buildings_{LIMIT}.csv")
STATE_FILE = os.path.join(project_root, "data", f"wikidata_buildings_state_{LIMIT}.txt")    # 途中再開用に最後のOFFSETを記録


HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "MyWikidataDownloader/1.0" # (your_email@example.com)"
}


# =====================================================================
def append_rows_to_csv(rows, csv_path: str):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            # w.writerow(["qid", "label"])
            w.writerow(["qid", "label", "class_label", "class_qid"]) # [memo] 全てのsubclassを1つのCSVにまとめる場合はどのクラス由来か分かるようにこれを使う. rowにclass_label, class_qidも追加する必要あり
        w.writerows(rows)


# =====================================================================


def listup_top_categories():
    """①カテゴリのリストアップ

    リストアップの方法はいくつかある。
    `notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb` 内の
    「①カテゴリのリストアップをいくつかの方法で試して比較する」でいくつか試している。
    そのうちの最も良かった方法 「### ⭐️ （= owl:Thing 直下）のクラス一覧」を採用した。
    """

    # （= owl:Thing 直下）のクラス一覧を取得する
    query = """PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?class ?label
WHERE {
GRAPH <http://dbpedia.org/resource/classes#> {
    ?class a owl:Class ;
        rdfs:subClassOf owl:Thing ;
        rdfs:label ?label .
    FILTER(lang(?label) = "en")
    FILTER(STRSTARTS(STR(?class), "http://dbpedia.org/ontology/"))
}
}
ORDER BY LCASE(STR(?label)) STR(?class)
    """
    # LIMIT 500

    print("\nDBpediaのowl:Thing直下のクラスをリストアップ中...")
    data = run_sparql(query)
    for b in data["results"]["bindings"][:20]:
        print(b["class"]["value"])
    print(f"... total {len(data['results']['bindings'])} classes found.")
    return


def listup_sub_classes_and_save(category_urls: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    選出したクラス（カテゴリ）から、各クラスの直下のサブクラスをリストアップする。

    ※ simpleな処理バージョン と 複雑な処理バージョンがあった。どちらも試したが、結果は同じだったためsimpleな処理バージョンを採用した。
    """
    # simpleな処理バージョン
    # memo: なぜか途中で止まることがあるので、resultの初期化をコメントアウトし，止まったcategoryから再開する
    print("\n各カテゴリの直下のdboサブクラスをリストアップ中...")
    print(f"len categories: {len(category_urls)}")

    query_base = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subClass ?label
WHERE {{
    ?subClass rdfs:subClassOf <{category}> .
    ?subClass rdfs:label ?label .
    FILTER(lang(?label) = "en")
}}
ORDER BY LCASE(STR(?label))
    """

    # 各カテゴリの直下のdboサブクラスを取得するクエリを生成
    result: Dict[str, List[Tuple[str, str]]] = {}

    for category in category_urls:
        print(f"カテゴリ: {category}")

        query = query_base.format(category=category)
        data = run_sparql(query)
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            print(f"\tカテゴリ {category}: 0 direct subclasses")

        rows = []
        for b in bindings:
            subClass_uri = b["subClass"]["value"]
            name = b["label"]["value"]
            rows.append((subClass_uri, name))
        result[category] = rows
        print(f"\tカテゴリ {category}: {len(rows)} direct subclasses")

    print(f"\nresult: {len(result)} categories with subclasses found.")

#     # 複雑な処理バージョン（ページネーション + リトライ + 途中再開機能付き）
#     ENDPOINT = "https://dbpedia.org/sparql"
#     ONTOLOGY_NS = "http://dbpedia.org/ontology/"
#     CLASSES_GRAPH = "http://dbpedia.org/resource/classes#"
#     def run_sparql(query: str, timeout: int = 60, max_retries: int = 4) -> dict:
#         headers = {
#             "Accept": "application/sparql-results+json",
#             "User-Agent": "dbpedia-subclass-fetch/0.1",
#         }
#         for i in range(max_retries):
#             try:
#                 r = requests.post(
#                     ENDPOINT,
#                     data={"query": query},
#                     params={"format": "application/sparql-results+json"},
#                     headers=headers,
#                     timeout=timeout,
#                 )
#                 # 429/503などはリトライしたい
#                 if r.status_code in (429, 503, 502, 504):
#                     time.sleep(1.5 * (i + 1))
#                     continue
#                 r.raise_for_status()
#                 return r.json()
#             except requests.RequestException as e:
#                 if i == max_retries - 1:
#                     raise
#                 time.sleep(1.5 * (i + 1))
#         raise RuntimeError("unreachable")

#     def fetch_direct_subclasses(category_url: str, lang: str = "en",
#                                 page_size: int = 1000, max_pages: int = 20
#                                ) -> List[Tuple[str, str]]:
#         parent_uri = f"<{category_url}>"

#         all_rows: List[Tuple[str, str]] = []
#         for page in range(max_pages):
#             offset = page * page_size
#             query = f"""
# PREFIX dbo: <http://dbpedia.org/ontology/>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# SELECT DISTINCT ?subClass ?label
# WHERE {{
#     ?subClass rdfs:subClassOf {parent_uri} .
#     ?subClass rdfs:label ?label .
#     FILTER(lang(?label) = "en")
# }}
# ORDER BY LCASE(STR(?label))
# OFFSET {offset}
# """    
#             data = run_sparql(query)
#             bindings = data.get("results", {}).get("bindings", [])
#             if not bindings:
#                 break

#             for b in bindings:
#                 child_uri = b["subClass"]["value"]
#                 name = b["label"]["value"]
#                 all_rows.append((child_uri, name))

#             # 念のため軽い間隔（公開endpointに優しい）
#             time.sleep(0.2)

#             # 最終ページ判定
#             if len(bindings) < page_size:
#                 break

#         return all_rows

#     def fetch_all(category_urls: List[str]) -> Dict[str, List[Tuple[str, str]]]:
#         result: Dict[str, List[Tuple[str, str]]] = {}
#         for c_url in category_urls:
#             if c_url.startswith(ONTOLOGY_NS):
#                 print(f"Fetching subclasses of {c_url}...")
                
#             rows = fetch_direct_subclasses(c_url, lang="en")
#             result[c_url] = rows
#             print(f"{c_url}: {len(rows)} direct subclasses")
#         return result
    

#     result = fetch_all(category_urls)

#     # 例：上位10件だけ表示
#     for c_url in category_urls:
#         print("\n===", c_url, "===")
#         for uri, name in result[c_url][:10]:
#             print(name, "->", uri)


    # ** 結果の保存 **
    # class - subclasses の構造を保存
    save_class_to_subclasses_path = os.path.join(project_root, "data", "maps", "dbpedia_classes_to_subclasses.json")
    os.makedirs(os.path.dirname(save_class_to_subclasses_path), exist_ok=True)
    with open(save_class_to_subclasses_path, "w") as f:
        json.dump(result, f, indent=2)

    # categories と，categories直下のsubclasses (result) を結合して保存
    all_classes = []
    for category in category_urls:
        if category in result.keys():
            # そのcategoryのsubClassesを取得できていれば，subClassesを追加. そのcategory自体は追加しない
            all_classes.extend(result[category])
        else:
            # そのcategoryのsubClassesがなかったのであれば，category自体を追加
            name = category.split("/")[-1]
            all_classes.append((category, name))
    print(len(all_classes), all_classes)

    # # 保存
    # path = os.path.join(project_root, "data", "dbpedia_selected_subclasses.json")
    # with open(path, "w") as f:
    #     json.dump(all_classes, f, indent=2)
    
    return



def map_classURL_to_QID(sub_classe_urls):
    """ クラスURLからQIDへのmappingを作成する
    """
    print("\nクラスURLからQIDへのmappingを作成中...")
    q = """PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?wikidata WHERE {
  <CLASS_URL>
    owl:equivalentClass ?wikidata .
}
"""
    # クラスのQIDを取得する
    QID_map = {}
    already_read = set()
    already_read.update(QID_map.keys())
    for class_url in sub_classe_urls:
        name = class_url.split("/")[-1].replace(" ", "_")
        print(name)
        if name in already_read:
            continue
        already_read.add(name)
        query = q.replace("<CLASS_URL>", f"<{class_url}>")
        data = run_sparql(query)
        print(data)
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

    print("\nクラスURL -> QIDのmapping:") 
    for name, QIDs in QID_map.items():
        print(f"- {name} -> {QIDs}")  

    # 保存
    path = os.path.join(project_root, "data", "dbpedia_to_wikidata_qid_map_20260607.json")   #"dbpedia_to_wikidata_qid_map_20260117.json")
    with open(path, "w") as f:
        json.dump(QID_map, f, indent=2)
    
    return QID_map
    


# query: Wikidata上で、あるクラス <<QID>> の1-hop下位クラスに属するエンティティのうち、英語Wikipedia記事があり、sitelinks数が多いものを上位 <<LIMIT_NUM>> 件取得する
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

def listup_proper_nouns_from_subclasses(sub_classes: List[Tuple[str, str]]):
    """選出したクラス（カテゴリ）から、各クラスの直下の固有名詞をリストアップする。
    """
    LIMIT = 1000
    SLEEP_SEC = 0.0 #2.0
    # STATE_FILE = os.path.join(project_root, "save", "wikidata_Things_offset.state") # 途中で中断しても再開できるようにオフセット(wikiから何個まで取得済みか)を保存
    OUT_CSV_DIR = os.path.join(project_root, "data", "dbpedia", f"wikidata_Things_childs_LIMIT{LIMIT}")
    os.makedirs(OUT_CSV_DIR, exist_ok=True)

    # QID_map読み込み
    path = os.path.join(project_root, "data", "dbpedia_to_wikidata_qid_map_20260607.json") #   "dbpedia_to_wikidata_qid_map_20260117.json")
    with open(path, "r") as f:
        QID_map = json.load(f)

    # 今回はoffsetは使わない．SPARQL_1HOP_MostSiteLinked毎にoffsetが変わるためいちいち保存したくないため．

    for class_url, class_label in sub_classes:    
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

            # 既にCSVがある場合、重複排除用に既存QIDを読み込む（巨大なら別方式推奨）
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
                
                try:
                    rows = fetch_page(SPARQL_1HOP_MostSiteLinked, LIMIT, offset, class_label, QID)
                except RuntimeError as e:
                    print(f"Error fetching page at offset={offset}: {e}. Moving to next class.")
                    break
                if not rows:
                    print("No more rows. Done.")
                    break
                pages_fetched += 1

                # 重複排除（QID基準）
                # new_rows = [(qid, label) for (qid, label) in rows if qid not in seen]
                new_rows = [(qid, label, class_label, class_QID) for (qid, label, class_label, class_QID) in rows if qid not in seen]
                for qid, _, _, _ in new_rows:
                    seen.add(qid)

                # append_rows_to_csv(new_rows, OUT_CSV)
                print(f"appending {len(new_rows)} rows to CSV (total seen {len(seen)})...")
                print(f"offset={offset} fetched={len(rows)} appended={len(new_rows)} total_seen={len(seen)}")

                offset += LIMIT
                # save_offset(offset)

                time.sleep(SLEEP_SEC)
    return




if __name__ == "__main__":
    # ** 1. 対象となるカテゴリ(クラス)をリストアップする **
    listup_top_categories()

    # ** 2. 対象となるカテゴリ(クラス)を人手で選んだ **
    # 実体を持つクラスを選ぶ。（クラス直下を見てあまり実態を伴ったものの集合ではない場合は弾く）
    category_urls = [
        "http://dbpedia.org/ontology/ArchitecturalStructure",
        "http://dbpedia.org/ontology/GrossDomesticProduct",
        "http://dbpedia.org/ontology/GrossDomesticProductPerCapita",
        "http://dbpedia.org/ontology/MeanOfTransportation",
        "http://dbpedia.org/ontology/Area",
        "http://dbpedia.org/ontology/Place",    
        "http://dbpedia.org/ontology/Medicine",
    ]
    # print("\n人手で選んだカテゴリ（クラス）:")
    # for cat in category_urls:
    #     print(f"“{cat.split('/')[-1]}”", end="，")
    # print("\n以上のカテゴリは、その下位に実体を伴ったクラスが多く含まれるものを対象に抽出した。")


    # # ** 3. 選出したクラスのの下位にある実体を伴ったクラスを抽出する + 保存 **
    # 目的: 大カテゴリ（category_urls）だけでなく、そのサブクラスからも、固有名詞を抽出したい。そのため、まずは大カテゴリの下位にあるサブクラスをリストアップする。
    # listup_sub_classes_and_save(category_urls)

    # 保存した，選択したサブクラスを読み込む
    path = os.path.join(project_root, "data", "maps", "dbpedia_classes_to_subclasses.json")
    with open(path, "r") as f:
        class_to_sub_classes = json.load(f)

    # 親クラス + sub_class のリストを作成
    sub_classes = []    # [(クラスURI, クラス名), ...] の形式
    for parent_class, subclasses in class_to_sub_classes.items():
        sub_classes.append((parent_class, parent_class.split("/")[-1])) # 親クラスも固有名詞抽出の対象にするため、親クラスも追加
        for sub_class in subclasses:
            sub_classes.append(sub_class)

    sub_class_urls = [url for url, name in sub_classes] # [class URL, ...] の形式

    # ** 4. 選出したクラスから、各クラス直下の固有名詞を取得する **
    # 準備: DBpediaのクラスURIからWikidata QIDへのmapping
    # (URLからの検索がうまくいかなかったため、QIDに変換してから検索することにした)
    QID_map = map_classURL_to_QID(sub_class_urls)

    # 各クラス（親クラス + サブクラス）から， 直下の固有名詞を取得しcsv保存する  [memo] 既に前のrepositoryで実施済みの処理であり、実行に時間がかかるため、コメントアウトを推奨
    # listup_proper_nouns_from_subclasses(sub_classes)


    # ** 5. 抽出結果の整理 **
    # [memo] `notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb` 内の 「③取得した固有名詞一覧を読み込み，数を大体揃える」は不要である可能性がある。
    # 理由は、この時は固有名詞の名前をそのまま使おうとしていたが、現在は架空の名前を使うため、名前の近さでいくつかの固有名詞を切り捨てる必要がないため。
    # また、カテゴリ毎の固有名詞の数も、後に適当に固有名詞をサンプリングして調整するため、ここで減らして数を揃える必要もないため。
    print("\n\n抽出した固有名詞の件数をクラスごとにカウント中...")
    path = os.path.join(project_root, "data", "dbpedia_to_wikidata_qid_map_20260607.json")
    with open(path, "r") as f:
        QID_map = json.load(f)
    
    class_propnoun_dir = os.path.join(project_root, "data", "dbpedia", f"wikidata_Things_childs_LIMIT{LIMIT}")
    sub_classes_with_data = [f.replace(".csv", "").replace("_", " ") for f in os.listdir(class_propnoun_dir) if f.endswith(".csv")]
    # print(sub_classes_with_data)

    whole_df = pd.DataFrame()
    for class_label in sub_classes_with_data:
        path = os.path.join(class_propnoun_dir, f"{class_label.replace(' ', '_')}.csv")
        df = pd.read_csv(path)
        # print(df.shape)
        # [memo] 元の`notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb` ではここに削除処理をおいていた
        whole_df = pd.concat([whole_df, df], ignore_index=True)

    # whole_dfをclass_labelでグループ化し，各グループごとに件数をカウントする
    grouped = whole_df.groupby('class_label').size().reset_index(name='counts')
    # 件数で降順にソートする
    grouped_sorted = grouped.sort_values(by='counts', ascending=False)
    print(f"\n各クラスの固有名詞の件数:")
    for _, row in grouped_sorted.iterrows():
        print(f"- {row['class_label']}: {row['counts']}件")







