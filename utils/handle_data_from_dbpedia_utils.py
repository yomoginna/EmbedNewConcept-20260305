
import os
import json
import pandas as pd
import random

project_root = os.path.join(os.path.dirname(__file__), "..") 
propNoun_dir = os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")


# *************** 固有名詞データ読み込み ***************
def loadProperNounData(propnoun_num_threshold=130, print_flag=False):
    """作成した固有名詞を読み込む関数。
    これは、初期化vec作成に使う固有名詞を収集するために使用する。

    Args:
        propnoun_num_threshold: 各カテゴリに属す固有名詞の最低数. この数に満たないカテゴリはfilterして返す
    """
    delete_list = ["year"] # year は、すべて2001, 2020, 2231 のような年の数字のみだったため削除。

    # *** dbpediaの各subclassファイルから、そのsubclassに属する固有名詞リストを取得する ***
    whole_df = pd.DataFrame()
    for category_file in os.listdir(propNoun_dir):
        if not category_file.endswith(".csv"):
            continue
        if category_file.replace(".csv", "") in delete_list:
            # 削除対象のカテゴリは読み込まない
            continue
        df = pd.read_csv(os.path.join(propNoun_dir, category_file))
        whole_df = pd.concat([whole_df, df], ignore_index=True)

    print(f"Total proper nouns collected: {len(whole_df)}")
    
    # *** fileter ***
    # 複数行に同じlabelがある場合は、そのlabelの行を削除する
    whole_df = whole_df.drop_duplicates(subset=['label'], keep=False)
    print(f"Total proper nouns after removing duplicates: {len(whole_df)}")

    # 中カテゴリごとの固有名詞の数をカウントし、dfにする
    category_count_df = whole_df.groupby("class_label").size().reset_index(name="count")

    # propnoun_num_threshold 以上の固有名詞があるカテゴリを抽出する
    # propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept
    filtered_category_count_df = category_count_df[category_count_df["count"] >= propnoun_num_threshold]
    print(f"{propnoun_num_threshold}以上の固有名詞があるカテゴリの数: {filtered_category_count_df.shape[0]}")

    # *** propnoun_num_threshold 以上の固有名詞があるカテゴリについて、属する固有名詞のリストを作成する ***
    filtered_categories = filtered_category_count_df["class_label"].tolist()
    filtered_category_to_properNouns_dict = {}
    for category in filtered_categories:
        properNouns = whole_df[whole_df["class_label"] == category]["label"].tolist()
        filtered_category_to_properNouns_dict[category] = properNouns

    # 表示
    if print_flag:
        print("Trainable categories and their proper nouns count:")
        for category, properNouns in filtered_category_to_properNouns_dict.items():
            print(f"Category: {category}, Proper Nouns Count: {len(properNouns)} {properNouns[:3]}...")
    return filtered_category_to_properNouns_dict


def load_prop_nouns(exclude_category=None, per_cat_limit=None):
    """dbpediaから収集した全ての固有名詞を収集. 
    loadProperNounDataとは異なり、カテゴリ毎の固有名詞数に基づくフィルタリングは行わない。単純に全ての固有名詞を収集する。
    """
    prop_nouns = []
    for category_file in os.listdir(propNoun_dir):
        if not category_file.endswith(".csv"):
            continue
        # もしexclude_categoryが指定されていれば、そのカテゴリの固有名詞は読み込まない
        category = category_file.removesuffix(".csv").replace("_", " ")
        if exclude_category == category:
            continue

        df = pd.read_csv(os.path.join(propNoun_dir, category_file))
        labels = df["label"].dropna().tolist()
        # 全部追加すると多すぎたので、各カテゴリからランダムに指定数(100や20など)個だけ追加することにする
        k = min(per_cat_limit or len(labels), len(labels))
        prop_nouns.extend(random.sample(labels, k))
    return prop_nouns



def loadConceptsForFictConcept():
    # 各概念毎に、架空の概念用の特徴の生成に成功した(=架空の概念用の)固有名詞を取得する
    category_to_concepts_for_fictconcept_path = os.path.join(project_root, "data", "generated_facts_in_wiki", "generated_concepts_map.jsonl")
    category_to_concepts_for_fictconcept = {}
    with open(category_to_concepts_for_fictconcept_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            category_to_concepts_for_fictconcept[data['category']] = data['successfully_generated_concepts']
    return category_to_concepts_for_fictconcept



def filterProperNounsWithWikiPage(properNouns_list, wiki_page_save_dir):
    """properNouns_listの中で、すでにwiki_page_save_dirに保存されているpropernounのみにフィルタリングする。
    これは、もう新しいwikiページを読み込みたくない場合に使用する。
    """
    filtered_properNouns = []
    for properNoun in properNouns_list:
        wiki_page_path = os.path.join(wiki_page_save_dir, f"{properNoun.replace(' ', '_')}.json")
        if os.path.exists(wiki_page_path):
            filtered_properNouns.append(properNoun)
    print(f"Filtered proper nouns with existing wiki pages: {len(filtered_properNouns)} / {len(properNouns_list)}")
    return filtered_properNouns




