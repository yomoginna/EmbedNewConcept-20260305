
import random
import os
import sys
# import math

# ===== Third-party =====
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
# from tqdm import tqdm
import torch
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import wandb

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)



def fix_seed(seed=0):
    """Fix random seed for reproducibility."""
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_gemma_model_version(model_size):
    model_size = int(model_size)
    if model_size in [2, 9]:
        model_version = 2
    elif model_size in [1, 4, 12]:
        model_version = 3
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    return model_version



# def save_mem_vec(model, memTokenIds, mem_save_path):
#     os.makedirs(os.path.dirname(mem_save_path), exist_ok=True)

#     # memTokenIdsの指定した要素の値(=weightのindex)の並び順のまま取り出され, 勝手にソートされることは無い
#     try:
#         vecs = (
#             model.model.embed_tokens.weight[memTokenIds]
#             .detach()
#             .to(torch.float32)
#             .cpu()
#             .numpy()
#         )
#     except AttributeError:
#         vecs = (
#             model.model.language_model.embed_tokens.weight[memTokenIds]
#             .detach()
#             .to(torch.float32)
#             .cpu()
#             .numpy()
#         )
#     np.save(mem_save_path, vecs)
# [memo] 上の処理を統一した
def save_mem_vec(model, memTokenIds, mem_save_path):
    os.makedirs(os.path.dirname(mem_save_path), exist_ok=True)

    embedding_layer = model.get_input_embeddings()

    # memTokenIdsの指定した要素の値(=weightのindex)の並び順のまま取り出され, 勝手にソートされることは無い
    vecs = (
        embedding_layer.weight[memTokenIds]
        .detach()
        .to(torch.float32)
        .cpu()
        .numpy()
    )

    np.save(mem_save_path, vecs)


# [memo] embedding_utils.pyに移動した
# def load_mem_vec(model, mem_save_path, memTokenIds):


def evaluateModel(model, tokenizer, evalInputs, evalOutputTexts, verbose=False):
    """
    Evaluate the model on the given inputs and outputs.
    """
    if len(evalInputs) == 0:
        raise ValueError("evalInputs is empty.")

    model.eval()
    # total_val_loss = 0.0
    # total_val_tokens = 0

    maxNewTokens = 10
    numCorrect = 0
    print('Evaluating on %d samples...'%len(evalInputs))
    with torch.no_grad():
        for i in range(len(evalInputs)):
            generation = model.generate(
                                    torch.LongTensor([evalInputs[i]]).to(model.device),
                                    max_new_tokens=maxNewTokens, 
                                    do_sample=False,
                                    repetition_penalty=1.05,
                                )[0]
            decodedGeneration = tokenizer.decode(generation)


            # if verbose and i // 2 == 0 or i == len(evalInputs) - 1: # ()必須。無いと優先順位が変わる。if verbose and i==0 or i==len(evalInputs)//2 or i==len(evalInputs)-1:
            if verbose and (i == 0 or i == len(evalInputs) // 2 or i == len(evalInputs) - 1):
                print('--- Sample %d ---'%i)
                # print('\tP:', decodedGeneration)
                # print('\tT:', evalOutputTexts[i])
                print(decodedGeneration.startswith(evalOutputTexts[i]))
            
            if decodedGeneration.strip().startswith(evalOutputTexts[i].strip()):
                # もし生成テキストが正解テキストと一致したら
                numCorrect += 1

    acc = numCorrect / len(evalInputs)
    return acc



def extract_probability_of_option_numbers(target_logits, tokenizer, num_options):
    """選択肢の数字の生成確率を抽出する
    Args:
        target_logits (torch.Tensor): モデルの出力ロジット
        tokenizer (transformers.AutoTokenizer): トークナイザ
        num_options (int): 選択肢の数
    Returns:
        log_prob_dicts (List[Dict[str, float]]): 各選択肢番号に対する生成log確率辞書のリスト. e.g. [{"1": -1.2, "2": -0.5}, {"1": -0.3, "2": -1.5}, ...]
        prob_dicts (List[Dict[str, float]]): 各選択肢番号の生成確率辞書のリスト
    """
    log_probs = F.log_softmax(target_logits, dim=-1) # logなので, 確率0~1.0は, マイナスor0 になる. 確率が小さいほどlogも小さくなる.
    probs = torch.exp(log_probs)  # log_probsを確率に変換
    # print(f"log_probs shape: {log_probs.shape}")  # (batch_size, vocab_size)

    # *** 選択肢のトークンIDを取得 ***
    # [memo] convert_tokens_to_ids() は「既にトークン化済みの文字列」をIDに変換する関数で、encode() は「普通の文字列を tokenizer の規則でトークン化してIDにする関数」
    # [memo] (両出力を見たところ実際には影響がなかったが念のため変更した。) Gemma / SentencePiece 系では "1" ではなく "▁1" のようなトークンになっている可能性があるため、tokenizer.encode()を使用して各選択肢番号のトークンIDを取得する方法に変更した。
    # number_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, num_options + 1)]
    number_token_ids = []
    for i in range(1, num_options + 1):
        token_ids = tokenizer.encode(str(i), add_special_tokens=False)  # "1" などの選択肢番号をトークン化してIDに変換。add_special_tokens=Falseにより、<bos> / <eos> / <pad> のような special token は追加されない

        if len(token_ids) != 1:
            raise ValueError(f"Option number {i} is not a single token: {token_ids}")

        number_token_ids.append(token_ids[0])



    # 各選択肢番号のトークンIDに対応するlog確率を取得
    token_log_probs_of_all_q = log_probs[:, number_token_ids]  # 各テスト問題に対する、各選択肢番号tokenのlog確率を取得
    token_prob_of_all_q = probs[:, number_token_ids]  # 各テスト問題に対する、各選択肢番号tokenの確率を取得
    # print(f"token_scores_lst shape: {token_scores_lst.shape}")  # (batch_size, num_options)

    # 各選択肢番号のlog確率を取得
    log_prob_dicts = []
    for token_log_probs_of_q in token_log_probs_of_all_q:
        num_2_log_prob = {}
        for i, score in enumerate(token_log_probs_of_q): # for num_token_id, score in zip(number_token_ids, token_log_probs_of_q):
            num_2_log_prob[f"{i + 1}"] = score.item() # num_2_log_prob[f"{tokenizer.convert_ids_to_tokens(num_token_id)}"] = score.item()
        log_prob_dicts.append(num_2_log_prob)
    
    prob_dicts = []
    for q_token_probs_of_q in token_prob_of_all_q:
        num_2_prob = {}
        for i, prob in enumerate(q_token_probs_of_q):
            num_2_prob[f"{i + 1}"] = prob.item()
        prob_dicts.append(num_2_prob)

    return log_prob_dicts, prob_dicts



# ********* calculate metrics **********
def calculate_metrics(y_pred_lst, y_true_lst):
    """各種scoreを計算する
    Args:
        y_pred_lst (list): モデルの予測結果リスト
        y_true_lst (list): 正解ラベルリスト
    Returns:
        dict: accuracy, precision, recall, F1スコア
    """
    # accuracy, precision, recall, F1の計算
    accuracy = accuracy_score(y_true_lst, y_pred_lst)
    precision = precision_score(y_true_lst, y_pred_lst, average='weighted', zero_division=0)
    recall = recall_score(y_true_lst, y_pred_lst, average='weighted', zero_division=0)
    f1 = f1_score(y_true_lst, y_pred_lst, average='weighted', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1
    }




# =========================
# pool_hs_type に応じた hidden state を取りvecを作成する関数
# # =========================
# @torch.no_grad()
# def extract_hidden_states(model, tokenizer, text_list, pool_hs_type, data_type, batch_size=8, layer_index=None, print_flag=False):
# [memo] → utils/embedding_utils.pyに移動した



# ******************************************* train functions ********************************************

def constructTrainSamples(concept_to_train_data_source, train_sample_format, conceptForFict2token_map, n_feat_in_a_sample=3, print_flag=False):
    train_samples = []
    for target_concept, train_data_list in concept_to_train_data_source.items():

        # 対応する空token名を取得. <unused0>など
        unused_token = conceptForFict2token_map[target_concept]
        
        for train_data in train_data_list:
            wiki_text_with_token = train_data['wiki_text_with_token']
            facts_with_token = train_data['facts_with_token']

            # factsの順番をランダムに入れ替える
            facts_with_token = random.sample(facts_with_token, len(facts_with_token))

            # *** featuresをn個ずつに分割して、1sampleあたり、summary1つ+特徴文n個の形式にする。最後の余りはそのまま1sampleにする。***
            train_sample_template = train_sample_format['train_sample']
            train_fact_sentence_template = train_sample_format['train_fact_sentence']

            for i in range(0, len(facts_with_token), n_feat_in_a_sample):
                fact_sentences = facts_with_token[i:i+n_feat_in_a_sample]
                fact_sentences_str = "\n".join([train_fact_sentence_template.format(fact_sentence=fs) for fs in fact_sentences])
                train_sample = train_sample_template.format(
                    unused_token=unused_token,
                    summary=wiki_text_with_token, 
                    fact_sentences=fact_sentences_str
                )
                train_samples.append(train_sample)
                if print_flag:
                    print(train_sample)
                    print("**********")
            
        # 各(架空)概念あたり1sample表示する
        if len(train_samples) > 0:
            print(f"Example train 1 sample for concept '{target_concept}':")
            print(train_samples[-1])
            print("**********")
        else:
            print(f"【Warning】no train samples created for concept '{target_concept}'.")

    print(f"Total {len(train_samples)} train samples created.")
    return train_samples


def encodeTrainSamplesWithTokenizer(train_samples, tokenizer, padTokenId, device):
    maxLength = 0
    trainingData = []
    evalInputs = []
    evalOutputTexts = []
    temp_i = 0
    for sample in train_samples:
        # tokenized = tokenizer(sample)
        inputIds = tokenizer.encode(sample, add_special_tokens=True)

        if maxLength < len(inputIds):
            maxLength = len(inputIds)

        trainingData.append(inputIds) # 次token予測タスクのデータ
        evalInputs.append(inputIds[:-2]) # rel毎にルールベースで文にしているため，relの後の単語も同一になる文が多い. そのため, 最後の2tokenのみ(<word> + '.')の予測で可否を評価することにする
        evalOutputTexts.append(sample)

        if temp_i < 5:
            print(f"inputIds: {inputIds}")
            print(f"decoded inputIds: {tokenizer.decode(inputIds)}")
            print(f"evalInputIds: {evalInputs[-1]}") # 最後に追加したevalInputIdsを表示
            print(f"decoded evalInputIds: {tokenizer.decode(evalInputs[-1])}")
            print()
            temp_i += 1

    # ** padding ** 
    # : [[132, 45, 67], [23, 78]] -> [[132, 45, 67, [PAD], [PAD]], [23, 78, [PAD], [PAD], [PAD]]]
    trainingData = [t_ids + [padTokenId] * (maxLength - len(t_ids)) for t_ids in trainingData]
    trainingData = torch.LongTensor(trainingData).to(device)
    
    indices = list(range(len(trainingData)))
    return trainingData, evalInputs, evalOutputTexts, indices




# def set_flag(tokenizer):
#     """
#     自動的に立てられるフラグを立てる。他にもflagを設定するかもしれないので関数化した。
#     """
#     # llama系はpad_tokenが設定されていないことがあるため，その場合はeos_tokenをpad_tokenに設定する
#     need_to_set_pad_token = False
#     if tokenizer.pad_token_id is None:
#         # tokenizer.pad_token_id = tokenizer.eos_token_id
#         tokenizer.pad_token = tokenizer.eos_token
#         need_to_set_pad_token = True
#     return need_to_set_pad_token


def set_tokenizer_and_model(tokenizer, model):
    """
    tokenizerとmodelの共通設定を行う関数
    """
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token   # llama系はpad_tokenが設定されていないことがあるため，その場合はeos_tokenをpad_tokenに設定する
        model.config.pad_token_id = tokenizer.pad_token_id



def construct_model_name_for_dirname(model_size, lr, trained_date, layer_idx, random_seed):
    model_version = get_gemma_model_version(model_size)

    model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}"
    if layer_idx is not None:
        print(f"Using layer index: {layer_idx}")
        model_name_for_dirname += f"-hidden_layer{layer_idx}"
    model_name_for_dirname += f"-seed{random_seed}"

    return model_name_for_dirname

