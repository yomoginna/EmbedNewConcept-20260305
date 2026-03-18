"""
geminiによる固有名詞推定結果をもとに, wikipageから生成した特徴文の中で, 元のconceptを特定するのに有用な特徴文を選別するコード.

uv run python src/select_useful_features_based_on_genai_guess_concept_results.py
"""
import json
import os
import sys

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)

print_flag = False #True False

# min_contribute_count回以上正解に貢献した特徴文だけを選別する場合の閾値
min_contribute_count = 1    # 2->2回以上正解に貢献したidだけをカウントする場合の閾値. 小さい値だと, ペアのもう一方の特徴が強いだけだったり, 他のconceptにも該当する特徴が残ってしまう可能性がある.
min_feature_num = 10 # 全relationで合計min_feature_num個以上の特徴文がある場合のみ保存

# concept名の特例. concept名をそのまま当てていなくても、このリスト内を推測していれば正解とする (例. "Adoration of the Kings" は "The Adoration of the Magi" とも呼ばれることがあるため、どちらも同じconceptとして扱う)
exception_of_concept_name = {
    "Adoration of the Kings": ["Adoration of the Magi", "The Adoration of the Magi"]
}


def main():
    # *** 準備 ***
    # 関係毎のmasked sentenceテンプレート読み込み
    with open(os.path.join(project_root, "data", "templates", "rel_to_maskedSentence.json"), "r", encoding="utf-8") as f:
        rel_to_maskedSentence = json.load(f)

    # wikipageから生成した特徴文データのdir
    wiki_feat_sentences_dir = os.path.join(project_root, "data", "generated_facts_in_wiki")

    # geminiによる固有名詞推定結果のdir
    guess_concept_result_dir = os.path.join(project_root, "data", "generated_guess_proper_noun_from_2facts")

    # このコードでfilteringした後に残った特徴文を保存する先のdir
    filtered_wiki_feat_sentences_save_dir = os.path.join(project_root, "data", "generated_facts_in_wiki_filtered")
    os.makedirs(filtered_wiki_feat_sentences_save_dir, exist_ok=True)



    # ********* concept毎に, geminiによる固有名詞推定結果を解析し, 有用な特徴文を選別する *********
    for filename in os.listdir(guess_concept_result_dir):
        if not filename.endswith(".jsonl"):
            continue
        concept = filename.split('.jsonl')[0].replace('_', ' ')
        print(concept)

        if concept in exception_of_concept_name.keys():
            concepts_for_comparison = exception_of_concept_name[concept]
        else:
            concepts_for_comparison = []

        # *** wikipageから生成した特徴文を読み込む ***
        with open(os.path.join(wiki_feat_sentences_dir, f"{concept.replace(' ', '_')}.json"), "r", encoding="utf-8") as f:
            wiki_feat_dt = json.load(f)

        # ** wiki_feat_dtのenの特徴文に, 通し番号を付ける **
        # sentenceが属すrelの特定のためにdt_with_idが必要. またidから文を特定するためid_to_sentenceも必要.
        # geminiによるconcept当て推論時と全く同じ手順で通し番号を付けるため, idとsentenceの組み合わせはguess_concept_result_dir内のidが指すsentenceと全く同じになる
        id = 0
        dt_with_id = {}
        id_to_sentence = {}
        for rel in wiki_feat_dt['parsed']['english'].keys():
            dt_with_id[rel] = []
            for s in wiki_feat_dt['parsed']['english'][rel]:
                dt_with_id[rel].append({
                    "id": id,
                    "sentence": s
                })
                id_to_sentence[id] = s
                id += 1


        # ********* 判定結果に応じてrecordを整理 *********

        # invalid_records = []  # json形式の適切さを判定する'ok'がFalseのものがないか確認
        correctly_identified_records = []  # 固有名詞を正しく特定できたもの
        allready_processed_pairs = [] # 既に処理した文ペアの(sentence_1, sentence_2)リスト. 生成時の同じ特徴文生成が原因の重複ペアを避けるために使用.

        with open(os.path.join(guess_concept_result_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                # print(record["sentence_1"], record["sentence_2"])

                # *** 重複ペアのスキップ処理 ***
                # 生成時の文の重複により, 同じsentence同士を比較している場合はskip
                if record["sentence_1"] == record["sentence_2"]:
                    continue

                # 生成時の文の重複により, 既に追加されたidが違ってもsentenceペアが同じものが既に追加されている場合はskip
                if  (record["sentence_1"], record["sentence_2"]) in allready_processed_pairs  or  \
                    (record["sentence_2"], record["sentence_1"]) in allready_processed_pairs:
                    if print_flag: print(f"\t【Skipped】({record['id_1']}, {record['id_2']}) Duplicate sentence pair detected. Sentences: '{record['sentence_1']}' , '{record['sentence_2']}'")
                    continue

                allready_processed_pairs.append( (record["sentence_1"], record["sentence_2"]) )


                # *** 判定結果に応じてrecordを 不正解recordセット, 正解recordセットに振り分ける ***
                # "ok": identifiable=False, proper_noun: "" のような不正なrecordでないことをcodeで確認した結果. 
                # if not record.get("ok", True):
                #     # 不正な結果の場合は不正解recordセットに追加
                #     invalid_records.append({
                #         "concept": concept,
                #         "record": record
                #     })
                # else:
                if record['response_info']['parsed']['identifiable']:
                    # * 2つの特徴文から元の固有名詞を特定できたもの:
                    print(record['response_info']['parsed']['proper_noun'].lower())
                    
                    if record['response_info']['parsed']['proper_noun'].lower() in [concept.lower()] + [c.lower() for c in concepts_for_comparison]:
                        # 特定した固有名詞が元のconceptと一致するのかについても確認してから, 元conceptを正しく推測するhintになった特徴文ペアとして追加
                        correctly_identified_records.append({
                            "record": record
                        })
                        if print_flag:
                            print(f"\t✅ Correctly identified the original proper noun. {record['response_info']['parsed']['proper_noun']} == {concept}")
                        print(f"\t✅ id: ({record['id_1']}, {record['id_2']})  Sentences: '{record['sentence_1']}' , '{record['sentence_2']}'\n")
                        continue
                    else:
                        print_invalid_record(record, print_flag)
                else:
                    # * 2つの特徴文から元の固有名詞を特定できなかったもの:
                    print_invalid_record(record, print_flag)


        print(f"Total correctly identified records: {len(correctly_identified_records)}")
        # if print_flag: print(f"Total invalid records: {len(invalid_records)}")


        # *** 『各idが何回正解に貢献したのか』/『relation毎に, min_contribute_count回以上正解に貢献したidが, 何個あったのか』をカウントする ***
        count = {}
        # id_to_sentence = {}
        for rec in correctly_identified_records:
            id_1 = rec["record"]["id_1"]
            id_2 = rec["record"]["id_2"]
            count[id_1] = count.get(id_1, 0) + 1
            count[id_2] = count.get(id_2, 0) + 1
            # id_to_sentence[id_1] = rec["record"]["sentence_1"]
            # id_to_sentence[id_2] = rec["record"]["sentence_2"]
        sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)

        # *** rel毎のid数カウントと, min_contribute_count回以上正解に貢献した特徴文を記録 ***
        rel_to_features = {}  # relationごとの, min_contribute_count回以上正解に貢献した特徴文+id情報リスト
        rel_count = {} # relationごとのカウント

        for id, cnt in sorted_count:
            # もし2回以上正解に貢献したidだけをカウントするならここのコメントアウトを外す
            if cnt < min_contribute_count:
                continue

            # relationを特定
            id_rel = None
            for rel, id_sent_list in dt_with_id.items():
                for item in id_sent_list:
                    if item["id"] == id:
                        id_rel = rel
                        break

            sentence = id_to_sentence[id]

            # * masked sentenceの[MASK]に入る部分を抽出 *
            masked_sentence = rel_to_maskedSentence.get(id_rel, None)
            if masked_sentence is None:
                print(f"Warning: No masked sentence found for relation '{id_rel}'")
                continue
            # [MASK]付き文の[MASK]以外の部分を，特徴文から引くことで, [MASK]に入る部分を抽出する. e.g. "It is [MASK]."の "It is " を, "It is a stadium located in Riffa."から引くと, "a stadium located in Riffa."が得られる.
            not_masked_part = masked_sentence.replace(".", "").replace("[MASK]", "") # e.g. "It is [MASK]." => "It is " (.も除く)
            mask_part = sentence.replace(".", "").replace(not_masked_part, "").strip()

            # 特徴文情報を保存
            rel_to_features.setdefault(id_rel, []).append({
                "id": id,
                "sentence": sentence,
                "masked_phrase": mask_part,
                "contribute_count": cnt
            })
            rel_count[id_rel] = rel_count.get(id_rel, 0) + 1
            # print(f"{cnt} count:  ID {id}, rel {id_rel}, '{id_to_sentence[id]}'")
        
        # rel_to_featuresのrelをrel_to_maskedSentenceのrelの順番にソートする
        rel_to_features_sorted = {}
        for rel in rel_to_maskedSentence.keys():
            if rel in rel_to_features:
                # idの順番もソートしておく
                rel_to_features_sorted[rel] = sorted(rel_to_features[rel], key=lambda x: x["id"])


        print(rel_count)
        print(rel_to_features_sorted)
        print()
        
        # 保存
        if len(sum(rel_to_features_sorted.values(), [])) > min_feature_num:  # 全relationで合計min_feature_num個以上の特徴文がある場合のみ保存
            with open(os.path.join(filtered_wiki_feat_sentences_save_dir, f"{concept}.json"), "w", encoding="utf-8") as f:
                json.dump(rel_to_features_sorted, f, ensure_ascii=False, indent=2)
        else:
            print(f"【Skipped saving】 Not enough features ({len(sum(rel_to_features_sorted.values(), []))} < {min_feature_num}), so not saved.\n")

    # print(f"Total invalid records: {len(invalid_records)}")



def print_invalid_record(invalid_record, print_flag=False):
    if print_flag:
        print(f"\t❌ Not identifiable (confidence: {invalid_record['parsed']['confidence']})")
        print(f"\t❌ id: ({invalid_record['id_1']}, {invalid_record['id_2']})  Sentences: '{invalid_record['sentence_1']}' , '{invalid_record['sentence_2']}'\n")



if __name__ == "__main__":
    main()