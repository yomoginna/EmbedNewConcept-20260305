"""
trainMemVec_fromXvec_gemma_wholeRun.pyで学習に使用するtrain dataを作成するためのスクリプト。
train dataのサンプルは、以下のような形式で保存する。
{
    "main_text": wikiページの本文テキスト,
    "summary": wikiページの要約テキスト,
    "facts": wikiから抽出された特徴文のリスト(filtering済み)
}
trainMemVec_fromXvec_gemma_wholeRun.py内で、ここで作成したファイルを元に、main_text or summaryと、factsから複数サンプルの学習データを構築することになる。

train data作成対象となるconceptを指定する場合:
```sh
uv run python src/construct_traindata.py --target_concepts "Adoration of the Kings" "Unlock!" "BattleFleet Mars"
```

wikiから抽出された特徴が一定以上存在するconceptすべてをtrain data作成対象とする場合:
```sh
uv run python src/construct_traindata.py
```

"""
import argparse
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.wikipedia_api_utils import extract_wiki_main_text, fetch_wikipedia_page
from utils.handle_data_from_dbpedia_utils import loadProperNounData

generated_facts_dir = os.path.join(project_root, 'data', 'generated_facts_in_wiki')
wiki_page_save_dir = os.path.join(project_root, 'data', 'wiki_pages')
os.makedirs(wiki_page_save_dir, exist_ok=True)
train_data_dir = os.path.join(project_root, 'data', 'train_data')
os.makedirs(train_data_dir, exist_ok=True)
train_target_concepts_path = os.path.join(project_root, 'config', 'target_concepts.json')


feat_num_threshold = 60     # wikiから抽出された特徴の数がこの数以上の固有名詞のみを学習データ作成の対象とする
propnoun_num_for_init_vec=100   # 初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept=30 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば30に設定した場合、各カテゴリで30個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。


def main(args):
    
    target_concepts = args.target_concepts


    # ****** 生成した固有名詞のリストを取得する ******
    # 現状では、"board game" と "Painting" についてのみ生成済み
    concepts_with_wikifeatures = []
    for file in os.listdir(generated_facts_dir):
        if file.endswith('.json'):
            # 特徴が60個以上取得できていた固有名詞のみ追加する
            with open(os.path.join(generated_facts_dir, file), 'r') as f:
                data = json.load(f)
            # 生成された(有効な)特徴数が feat_num_threshold 未満ならスキップする
            total_features = sum(data['parsed']['english'].values(), [])
            total_features = [feat for feat in total_features if feat.lower() != "unknown"] # featが unknown や Unknown などの意味のない特徴であれば削除する
            if len(total_features) < feat_num_threshold:
                continue
            concept = ' '.join(file.replace('.json', '').split('_'))
            concepts_with_wikifeatures.append(concept)

    # target_conceptsをアルファベット順にソート
    concepts_with_wikifeatures.sort()
    # print(f"Concepts with wiki features: {concepts_with_wikifeatures}")


    # ****** 学習データ作成対象の固有名詞を決定する ******
    if target_concepts is None:
        # conceptsの指定がなければ、wikiから抽出した特徴が一定以上存在する固有名詞すべてを対象とする
        target_concepts = concepts_with_wikifeatures
    else:
        # target_conceptsが指定されている場合は、指定されたもののうち、wikiから抽出した特徴が一定以上存在する固有名詞のみを対象とする
        target_concepts = [concept for concept in target_concepts if concept in concepts_with_wikifeatures]

    print(f"Target concepts: {target_concepts}")

    # ****** 学習データに含めるため、各target_conceptに対応するwikipageを取得する ******
    for target_concept in tqdm(target_concepts, desc="Fetching Wikipedia pages"):
        # 既に保存されていればスキップする
        save_path = os.path.join(wiki_page_save_dir, f"{target_concept.replace(' ', '_')}.json")
        if os.path.exists(save_path):
            print(f"Wikipedia page for concept '{target_concept}' already exists. Skipping.")
            continue

        # wikipageのテキストをapiで取得する
        wiki_info = fetch_wikipedia_page(target_concept, lang="en")
        if wiki_info["exists"] == False:
            print(f"Wikipedia page for concept '{target_concept}' DOES NOT exist. Skipping generation.")
            continue
        # 本文を切り出す
        main_text = extract_wiki_main_text(wiki_info['text'])
        wiki_info['text'] = main_text

        # 保存
        with open(save_path, "w") as f:
            json.dump(wiki_info, f, ensure_ascii=False, indent=4)



    # ****** wikipageのテキストと、生成した特徴をもとに、学習データのサンプルを作成する ******
    for target_concept in tqdm(target_concepts, desc="Creating training data samples"):
        # wikiページのテキストを読み込み
        with open(os.path.join(wiki_page_save_dir, f"{target_concept.replace(' ', '_')}.json"), "r") as f:
            wiki_info = json.load(f)

        # wikiを元に生成された特徴データを読み込み
        with open(os.path.join(generated_facts_dir, f"{target_concept.replace(' ', '_')}.json"), "r") as f:
            generated_facts = json.load(f)
        # unknown な特徴をfilteringする
        fact_sentences = []
        for rel, feats in generated_facts['parsed']['english'].items():
            for feat in feats:
                if feat.lower() in ["unknown", '不明']:
                    continue
                fact_sentences.append(feat)

        data_sample = {
            "main_text": wiki_info['text'], 
            "summary": wiki_info['summary'],
            "facts": fact_sentences
        }
        """
        以下2つは、train dataのsampleのtemplateにおける[SUMMARY]部分に入る情報．main_textは[FACT]として追加できる特徴文の量に対して長過ぎるため、現状ではsummaryを使用している。
            main_text: wikiページの本文テキスト
            summary: wikiページの要約テキスト
        以下は、[FACT]として追加できる特徴文の例の一覧。train用の.pyファイル内でshuffleし、1sampleに何個の特徴文を含めるかを指定した数に応じて柔軟にtrain dataを作成するため、ここではsample毎に分けて保存するのではなく、factsのリストとしてまとめて保存する。
            facts: wikiから抽出された特徴文のリスト(filtering済み)
        """

        with open(os.path.join(train_data_dir, f"{target_concept.replace(' ', '_')}.json"), "w") as f:
            json.dump(data_sample, f, ensure_ascii=False, indent=4)

    # ****** Mid class (category) - target concepts のmapを作成して保存する ******
    # 全てのカテゴリ・固有名詞リスト の辞書を読み込む (重複等のfiltering済み)
    filtered_category_properNouns_dict = loadProperNounData(
        propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept,
        print_flag=True
    )
    # target_concepts が属するカテゴリを、target_concept - category のmapから抜き出す               
    target_concepts_set = set(target_concepts)
    category_to_targetConcepts_dict = defaultdict(list)
    for category, properNouns in filtered_category_properNouns_dict.items():
        matched = target_concepts_set & set(properNouns)
        category_to_targetConcepts_dict[category].extend(matched)

    # 保存
    with open(train_target_concepts_path, "w") as f:
        json.dump(category_to_targetConcepts_dict, f, ensure_ascii=False, indent=4)

    print("Training data construction completed.")
    

        





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_concepts", nargs="+", help="List of concepts to fetch Wikipedia pages for")
    args = parser.parse_args()
    main(args)
