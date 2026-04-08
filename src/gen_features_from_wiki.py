import pandas as pd
import json
import re
import os
import sys
import random
import time
import argparse
from tqdm import tqdm
import wikipediaapi
from dotenv import load_dotenv
load_dotenv() # API keyの読み込み

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.llm_generation_utils import gen_with_google_genai_api, gen_with_openai_api

# from openai import OpenAI
# client = OpenAI()

# gptは少し高いのでgeminiに変更。wikiからの特徴抽出タスクは複雑な思考を必要としないため、軽量のモデルで十分と判断。
from google import genai
client = genai.Client()


# 定数
propnoun_num_for_init_vec = 100 # 1カテゴリあたりで、初期化vec(平均vec)の作成に使う概念数
propnoun_num_for_new_concept = 50 # 1カテゴリあたりで、新規概念の元にする概念数
model = "gemini-2.5-flash-lite"
max_retries = 1
feat_num_threshold = 60 # 1概念あたりで、生成された特徴の数がこの数以上であれば生成成功とみなす
wiki_word_num_threshold = 500 # 十分な情報量のあるwikipageに対してのみ生成を行うための、wikipageの本文の単語数の閾値
propnoun_num_for_new_concept = 50
propnoun_num_for_init_vec = 100
temperature = 0.2
topP = 0.8
RANDOM_SEED = 42
delay = 0.5 # 0.3
print_results = True  # get_concept_list_with_complete_cos_similar_termsの結果を表示するかどうか

gen_result_savedir = os.path.join(project_root, "data", "generated_facts_in_wiki")
os.makedirs(gen_result_savedir, exist_ok=True)
generated_concepts_save_filepath = os.path.join(gen_result_savedir, "generated_concepts_map.jsonl")
rel_to_maskedSentence_path = os.path.join(project_root, 'data', 'templates', 'rel_to_maskedSentence.json')

def main(args):
    
    # 引数
    target_categories = args.target_categories
    print(f"Target categories specified: {target_categories}. Will generate features only for these categories.")
    

    # ************* 固有名詞読み込み *************
    delete_list = ["year", "Election"] # year は、すべて2001, 2020, 2231 のような年の数字のみだったため削除。Electionも実体ではない

    whole_df = pd.DataFrame()
    for filename in os.listdir(os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000")):
        if filename.replace(".csv", "") in delete_list:
            # 削除対象のカテゴリは読み込まない
            continue
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(project_root, "data", "dbpedia", "wikidata_Things_childs_LIMIT1000", filename))
            whole_df = pd.concat([whole_df, df], ignore_index=True)
    print(f"Total proper nouns collected: {len(whole_df)}")
    
    # *** 複数行に同じlabelがある場合は、そのlabelの行を削除する ***
    # - subclassのsubclassを取っているものもあるため、上位のsubclassとその下位のsubclassの両方に同じ固有名詞が属している心配がある。
    # 複数行に同じlabelがある場合は、そのlabelの行を削除する
    whole_df = whole_df.drop_duplicates(subset=['label'], keep=False)
    print(f"Total proper nouns after removing duplicates: {len(whole_df)}")

    # *** 属する固有名詞が一定数以上ある中カテゴリを、実験対象として選ぶ ***
    # 中カテゴリごとの固有名詞の数をカウントし、dfにする
    category_count_df = whole_df.groupby("class_label").size().reset_index(name="count")

    propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept
    filtered_category_count_df = category_count_df[category_count_df["count"] >= propnoun_num_threshold]
    print(f"{propnoun_num_threshold}以上の固有名詞があるカテゴリの数: {filtered_category_count_df.shape[0]}")
    
    if target_categories:
        target_categories = [cat for cat in target_categories if cat in filtered_category_count_df["class_label"].tolist()]
        print(f"Target categories specified: {target_categories}. Will generate features only for these categories.")
    else:
        target_categories = filtered_category_count_df["class_label"].tolist()
        print("No target categories specified. Will generate features for all categories.")


    # ************* wiki page の本文から特徴抽出 *************
    # - 各カテゴリあたり、固有名詞50個(propnoun_num_for_new_concept)について特徴を抽出する。
    # - まずカテゴリ毎に固有名詞をシャッフルする。
    # - そして、最初の固有名詞から順番に特徴を抽出していく。
    # - 特徴が60個(propnoun_num_for_new_concept+10)取れなかった固有名詞は捨て、次の固有名詞に移る。
    # - これを繰り返し、特徴が60個取れた固有名詞を50個集める。
    # - 固有名詞が50個集まった時点で、そのカテゴリの特徴抽出は終了とする。

    print("=== Generating factual feature sentences from Wikipedia articles ===")

    # *** 準備 ***
    # prompt と jsonschema を読み込み
    print("Preparing prompts and schemas...")
    print("Preparing prompts and schemas...")
    if model.startswith("gemini"):
        prompt_schema_path = os.path.join(project_root, "data", "prompt", "gen_wiki_fact_extraction_for_googleai_schema.json")
    elif model.startswith("gpt"):
        prompt_schema_path = os.path.join(project_root, "data", "prompt", "gen_wiki_fact_extraction_for_openai_schema.json")
    prompt_text_path = os.path.join(project_root, "data", "prompt", "gen_wiki_fact_extraction_baseprompt.txt")

    with open(prompt_schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(prompt_text_path, "r", encoding="utf-8") as f:
        gen_triplet_prompt_base = f.read()


    # *** 既に生成済みのconceptをリストアップ ***
    # 生成済みのconceptについては生成をスキップするため、事前に確認しておく
    # print("Checking already generated concepts...")
    generated_concepts = []
    for filename in os.listdir(gen_result_savedir):
        if filename.endswith(".json"):
            concept = filename.split('.json')[0]  # 拡張子を除いた部分を取得
            concept = concept.replace('_', ' ')  # ファイル名のアンダースコアをスペースに変換
            generated_concepts.append(concept)
    print(f"Already generated concepts: {generated_concepts}")

    # *** 生成ループ: target_categoriesの各カテゴリについて、Wikipediaから特徴抽出を行う ***
    for target_category in target_categories:
        # 再現性のためにシードを固定 
        # - target_categoriesに複数カテゴリを指定した場合も生成対象のconceptが同じになるよう、seedはカテゴリ内のループ開始時に設定する
        random.seed(RANDOM_SEED)

        # target_categoryに属する固有名詞を全てリストアップし、シャッフルする
        target_concepts = whole_df[whole_df["class_label"] == target_category]["label"].tolist()
        target_concepts = sorted(target_concepts)
        target_concepts = random.sample(target_concepts, k=len(target_concepts))


        # * カテゴリ内の各固有名詞について、wiki page取得・特徴生成をループする
        gen_success_count = 0   # 生成成功した概念の数 (生成された特徴の数が feat_num_threshold 以上であった概念の数)
        total_gen_count = 0     # 生成を試みた概念の総数 (生成成功・失敗に関わらず)
        gen_success_concepts = [] # 生成成功した概念のリスト (生成された特徴の数が feat_num_threshold 以上であった概念のリスト)
        generated_concepts = [] # 生成成功・失敗に関わらず、生成を試みた概念のリスト
        for target_concept in target_concepts:
            # 既に生成済みのconceptであればスキップする
            if target_concept in generated_concepts:
                print(f"Concept '{target_concept}' ALREADY has GENERATED factual feature sentences. Skipping generation.")
                continue
            # 十分な数の固有名詞について特徴が生成できたら、このカテゴリの生成は終了する
            if gen_success_count >= propnoun_num_for_new_concept:
                print(f"REACHED the target number of successfully generated concepts ({propnoun_num_for_new_concept}). Stopping generation for this category.")
                break

            # wikipage取得・生成・保存
            total_gen_count += 1
            print(f"Generating factual feature sentences for concept: {target_concept}")
            gen_succeeded = gen_and_save_wiki_fact_extraction_for_concept(
                target_concept=target_concept,
                feat_num_threshold=feat_num_threshold,
                wiki_word_num_threshold=wiki_word_num_threshold,
                prompt_template=gen_triplet_prompt_base,
                schema=schema,
                gen_result_savedir=gen_result_savedir,
                model=model,
                temperature=temperature,
                topP=topP,
                max_retries=max_retries
            )
            print(f"\tGeneration result for {target_concept}: {gen_succeeded}")
            gen_success_count += gen_succeeded
            if gen_succeeded:
                gen_success_concepts.append(target_concept)
            generated_concepts.append(target_concept)

        print(f"{gen_success_count}/{total_gen_count} concepts successfully generated with factual feature sentences for the category {target_category}.")

        # どのカテゴリでどの概念を生成できたのかを記録する. jsonl形式で保存 (1行に1カテゴリ分)
        record = {
            "category": target_category,
            "successfully_generated_concepts": gen_success_concepts,
            "generated_concepts": generated_concepts,
            "generation_success_count": gen_success_count,
            "generation_attempt_count": total_gen_count,
            "generation_success_ratio": gen_success_count / total_gen_count if total_gen_count > 0 else 0.0,
        }
        with open(generated_concepts_save_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def gen_and_save_wiki_fact_extraction_for_concept(
        target_concept, 
        feat_num_threshold, 
        wiki_word_num_threshold, 
        prompt_template, 
        schema, 
        gen_result_savedir, 
        model="gemini-2.5-flash-lite", 
        temperature=0.2, 
        topP=0.8,
        max_retries=1
    ):
    """
    ある概念に対して、Wikipediaのページを取得し、その情報をもとに、その概念の特徴文を生成し、保存する。
    Generate factual feature sentences for a given concept using Wikipedia information.
    Args:
        target_concept (str): The concept for which to generate factual features.
        feat_num_threshold (int): The minimum number of valid features required for successful generation.
        wiki_word_num_threshold (int): The minimum number of words required in the Wikipedia text for successful generation.
        prompt_template (str): The template for the generation prompt, containing placeholders for the target concept and Wikipedia text.
        schema (dict): The JSON schema that the generated response should adhere to.
        gen_result_savedir (str): The directory where the generated results should be saved.
        model (str): The model to use for generation. Default is "gemini-2.5-flash-lite".
        temperature (float): The temperature parameter for generation. Default is 0.2.
        topP (float): The topP parameter for generation. Default is 0.8.
        max_retries (int): The maximum number of retries for API calls. Default is 1.
    Returns:
        bool: True if generation is successful (i.e., the number of valid features meets the threshold), False otherwise.

    もし生成された特徴の数(unkonwn除く)がfeat_num_threshold未満であっても、生成結果自体は保存する。
    """

    with open(rel_to_maskedSentence_path, 'r') as f:
        rel_to_maskedSentence = json.load(f)


    # wikipageのテキストをapiで取得する
    wiki_info = fetch_wikipedia_page(target_concept, lang="en")
    if wiki_info["exists"] == False:
        print(f"Wikipedia page for concept '{target_concept}' DOES NOT exist. Skipping generation.")
        return False
    main_text = extract_wiki_main_text(wiki_info['text'])
    # print(main_text)
    
    if len(main_text.split()) < wiki_word_num_threshold:
        # もし十分な情報量(本文の文字数)がない場合は、生成をスキップする
        print(f"Concept '{target_concept}' has insufficient Wikipedia text ({len(main_text.split())} words). Skipping generation.")
        return False

    # prompt構築
    prompt = prompt_template.replace("<target_concept>", target_concept)
    prompt = prompt.replace("<wikipedia_text>", main_text[:10000])  # wikiページ本文は10000文字までに制限 (例. "Portsmouth Harbour"の本文は文字数9498, 単語数は1538)

    # 生成
    if model.startswith("gemini"):
        response = gen_with_google_genai_api(client, prompt, schema, temperature, topP, model, max_retries)
        time.sleep(delay) # APIに過度にリクエストを送らないように、生成と生成の間に少し待機時間を入れる
        if response is not None:
            # 辞書内で、relと特徴が整合していない場合があるので、その関係templateに適したrelに特徴を入れ直す
            parsed_organized = organize_response(response.parsed, rel_to_maskedSentence)
            record = {
                "response_id": response.response_id,
                "model_version": response.model_version,
                "text": response.text,
                "parsed": parsed_organized, # response.parsed,
                "usage_metadata": {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count,
                },
            }
            # 生成された(有効な)特徴数が feat_num_threshold 以上であれば生成成功とみなす
            total_features = sum(response.parsed['english'].values(), [])
            total_features = [feat for feat in total_features if feat.lower() != "unknown"] # featが unknown や Unknown などの意味のない特徴であれば削除する
            if len(total_features) >= feat_num_threshold:
                gen_succeeded = True
            else:
                gen_succeeded = False
        else:
            record = {
                "error": "Failed to generate a valid response after maximum retries.",
                "prompt": prompt,
                "schema": schema,
                "model": model,
                "max_retries": max_retries,
            }
            gen_succeeded = False

    elif model.startswith("gpt"):
        response = gen_with_openai_api(prompt, schema, temperature, topP, model, max_retries)
        pass # [WIP] ここではまだgpt-5-miniでの生成は試さないため、実装は保留中
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    # 保存
    filename = f"{target_concept.replace(' ', '_')}.json"
    with open(os.path.join(gen_result_savedir, filename), "w", encoding="utf-8") as f:
        # f.write(json.dumps(record, ensure_ascii=False) + "\n") # jsonl形式で1fileに追加し続ける場合 ('w'ではなく'a'にする必要あり)
        json.dump(record, f, ensure_ascii=False, indent=4)

    return gen_succeeded


def organize_response(parsed, rel_to_maskedSentence):
    """
    辞書内で、relと特徴が整合していない場合があるので、その関係templateに適したrelに特徴を入れ直す
    例えば、"has part"の特徴が "is part of"のrelの下に入っている場合など。
    """
    organized = {}
    organized['english'] = {rel: [] for rel in rel_to_maskedSentence.keys()}
    features = sum(parsed['english'].values(), [])
    for current_rel, features in parsed['english'].items():
        if current_rel == 'IsA':
            # IsAのrelは、KnownAsやUsedForなど他のrelのtemplateとの被りが多いこと、また間違いにくいことから、変更なしで要素をそのまま戻す
            organized['english'][current_rel].extend(features)
            continue
        for fact in features:
            if fact.lower() in ["unknown", "不明"]:
                # 変更なしで要素をそのまま戻す
                organized['english'][current_rel].append(fact)
                continue
            for new_rel, template in rel_to_maskedSentence.items():
                if new_rel == 'IsA':
                    # 'IsA'は判定から除外. 既にcurrent_relが'IsA'のときは、変更なしで要素をそのまま戻している.
                    continue
                mask_pattern = re.escape(template).replace("\\[MASK\\]", "(.+)")  # [MASK]をキャプチャグループに置換
                match = re.match(mask_pattern, fact)
                if match:
                    # maskFeat = match.group(1)  # キャプチャグループから[MASK]部分を抽出
                    organized['english'][new_rel].append(fact)
                    if current_rel != new_rel:
                        print(f'!!! Changed: {current_rel} -> {new_rel}: {fact}')
                    break
            # break されなかったときだけ実行される
            else:
                # 該当するrelがない場合、変更なしで要素をそのまま戻す
                organized['english'][current_rel].append(fact)
                print(f"no match Fact: '{fact}'")
    organized['japanese'] = parsed['japanese']  # 日本語版はただの和訳であり、templateは用意していない

    if len(sum(organized['english'].values(), [])) != len(sum(parsed['english'].values(), [])):
        print("Warning: The total number of features after organizing does not match the original count.")
    return organized



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

    if not page.exists():
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



# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_categories", nargs="+", default=None, help="The target categories for which to generate features. If not specified, features will be generated for all categories.")
    args = parser.parse_args()
    main(args)