
# ===== Standard library =====
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
import random
import re
import sys
import time

# ===== Third-party =====
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from plotnine import labels
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
import wandb

# ===== Runtime config =====
transformers_logging.set_verbosity_error()

project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
# project_root = os.environ["HOME"] # [memo] genkaiを使う場合. "/singularity_home/project/EmbedNewConcept/src/trainMemVec_fromXvec_gemma.py"
sys.path.append(project_root)
print("Project root:", project_root)

from utils.gemma_train_and_test_utils import fix_seed, save_mem_vec, evaluateModel
from utils.handle_data_from_dbpedia_utils import filterProperNounsWithWikiPage, loadProperNounData #, loadConceptsForFictConcept
from utils.initialize_embedding_layer_utils import EmbedInitializer
from utils.wandb_utils import set_wandb_env

# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # [memo] genkaiを使う時はコメントアウト!! -> 今はargsで指定している。argsを指定しなければ、CUDAについては何も指定しない。
n_feat_in_a_sample = 3  # 学習データの1サンプル = summary(wiki中の本文 or summary, 今回はsummaryを使用) + n_feat_in_a_sample個の特徴文
propnoun_num_for_init_vec=100   #  初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept = 50 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば50に設定した場合、各カテゴリで50個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。

global BATCH_SIZE

wiki_page_save_dir = os.path.join(project_root, 'data', 'wiki_pages')
dont_get_new_wiki_flag = True # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.

# 環境変数読み込み
load_dotenv(os.path.join(project_root, ".env"))
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# from huggingface_hub import login
# access_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# login(access_token)





# *************************** func ***************************

class GradZeroHook:
    """ 特定の行(=tokenのidx)の勾配を0にするフック
    学習対象外のembedding行の勾配は全て0にするために用いる
    token_id ごとに True / False を直接切り替えることはできないため, この勾配フックを使って、学習したい token_id 以外の勾配を 0 にする
    * どのモデルでも共通のはず
    """
    def __init__(self, embeddingsToKeep):
        self.embeddingsToKeep = embeddingsToKeep # どの行(=tokenのidx)の勾配を0にするかはGradZeroHook class自身が持っておく必要があるため，インスタンス生成時にselfに保存する

    def setGradsToZeroHook(self, grad):
        grad = grad.clone() # 新しいメモリ上に値をコピー. これは，元の勾配テンソルを直接変更しないようにするための安全策。PyTorchの勾配計算では，元のテンソルが他の場所で使われている可能性があるため，直接変更すると予期せぬ副作用が発生することがある。clone()を使うことで，元のテンソルを保護しつつ，新しいテンソル上で安全に操作ができるようになる。
        grad[self.embeddingsToKeep] = 0.0 # 指定された行の勾配を0にする
        return grad


def prepareGemmaModel(
        model_name,
        save_mem_dir,
        model, 
        tokenizer,
        train_token2tokenid, 
        init_vec_type, 
        pool_hs_type,
        category_to_concepts_for_vec, 
        category2initoken_ids,
        seed,
        layer_idx=False,
        print_flag=False
    ):
    """ Gemmaモデルを埋め込み層のみ学習できるように準備する
    1. model準備
    2. embedding以外のパラメータを凍結
    3. 指定のトークン以外は勾配が0.0になるようにフックを設定
    4. 損失関数準備

    Args:
        model: HuggingFaceのモデルオブジェクト
        num_trainTargetTokens: 学習対象とする特殊トークンの数 (架空のobjの数) [memo] llama, qwenのコードにはこれを追加していなかった。勾配をfreezeするtoken数をなるべく多くしてメモリ使用量を抑えるために導入
        init_vec_type: memory vectorの初期化方法。zeroまたはuniform, または語句. zero->0vec, uniform->一様分布, 語句->指定の語句の埋め込みベクトルで初期化, 数字->指定のコサイン類似度で近い語句のベクトルで初期化
        pool_hs_type: 隠れ状態をプーリングする方法。["eos", "last_token", "mean_pool"] のいずれか。init_vec_typeが 'category_centroid_by_hidden_state_mean' の場合に使用
        category_to_concepts_for_vec: カテゴリごとのvec初期化に使用する概念のリスト。init_vec_typeが 'category_COG' の場合に使用
        layer_idx: 隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。init_vec_typeが 'category_centroid_by_hidden_state_mean' の場合に使用
        category2initoken_ids: カテゴリごとの初期化トークンIDのリスト。init_vec_typeが 'category_COG' の場合に使用
    memo:
    * Qwenとは違い，special_tokenが最初から用意されているため，tokenizerの拡張は不要．
        * よって，tokenizerもここで準備する
    * special_tokenは全て初期化し, 学習可能な状態にする
    """
    print('Prepare model')
    
    if model is not None:
        # dataの状態だけprintするために、model=Noneとすることがある。その場合はmodel関連の処理はskipする。
        # *** embedding層以外の勾配を凍結 ***
        for param in model.parameters():
            param.requires_grad = False

        # *** embedding層内でも，対象外のtoken(reserved_special_token以外)の勾配を凍結 ***
        # <unused0>から順にnum_trainTargetTokens個のtokenIDを学習対象とし、この予約済み特殊token以外は勾配が0.0になるようにhookをかける. 
        # 学習可能tokenをなるべく少なく制限する理由は、gemma3のtokenizerに6242個もunused tokenが存在しており、これら全てを学習対象にしてしまうとモデルサイズが大きくなりすぎてしまうため。
        

        # [memo] ここでembed_tokens.weightが見つからないというerror。1bでは大丈夫だったのになぜ -> gemma-3-1bは言語モデルのみだが、4b以降はvision encoderが含まれているため、(vision_tower)と(language_model)の2つの大きなモジュールがmodelの直下に存在している。そのため、embedding層にアクセスするにはmodel.model.language_model.embed_tokens.weightとなる。model.model.embed_tokens.weightが見つからなかった時にprint(model.model)を表示したことで判明。
        # token_id ごとに True / False を直接切り替えることはできない. 勾配フック(GradZeroHook)を使って、学習したい token_id 以外の勾配を 0 にする
        try:
            # * 1b以下はvision_towerがないため、従来通りでOK
            model.model.embed_tokens.weight.requires_grad = True
        except:
            # print(model.model)
            # raise ValueError("モデルに embed_tokens.weight が見つかりません。Gemma3ベースのモデルを指定していることを確認してください。")
            # * 4b以上はvision_towerがあるため、language_modelを経由してアクセスする
            model.model.language_model.embed_tokens.weight.requires_grad = True
        

        # # *** embedding層内でも，対象外のtoken(reserved_special_token以外)の勾配を凍結 ***
        # # <unused0>から順にnum_trainTargetTokens個のtokenIDを学習対象とし、この予約済み特殊token以外は勾配が0.0になるようにhookをかける. 
        # # 学習可能tokenをなるべく少なく制限する理由は、gemma3のtokenizerに6242個もunused tokenが存在しており、これら全てを学習対象にしてしまうとモデルサイズが大きくなりすぎてしまうため。
        trainTokenIds = list(train_token2tokenid.values())
        
        try:
            # 上のtry-except同様の理由(vision_towerの有無)で処理を分ける.
            embeddingsToKeep = [i for i in range(model.model.embed_tokens.weight.shape[0]) if i not in trainTokenIds]
            gzh = GradZeroHook(embeddingsToKeep)
            model.model.embed_tokens.weight.register_hook(gzh.setGradsToZeroHook) # 登録したフックは勾配計算直後に呼び出される．関数の返り値がNoneなら元の勾配が，返り値がテンソルならそのテンソルが新しい勾配として使われる．
        except:
            embeddingsToKeep = [i for i in range(model.model.language_model.embed_tokens.weight.shape[0]) if i not in trainTokenIds]
            gzh = GradZeroHook(embeddingsToKeep)
            model.model.language_model.embed_tokens.weight.register_hook(gzh.setGradsToZeroHook)



    # *** 訓練対象tokenの埋め込みを初期化 ***
    train_target_category_lst = sorted(category2initoken_ids.keys()) # target_category_lst をアルファベット順にsort
    print(f"train_target_category_lst: {train_target_category_lst}")

    embed_initializer = EmbedInitializer(
        model_name,
        save_mem_dir,
        init_vec_type, 
        train_target_category_lst,
        propnoun_num_for_init_vec, 
        model, 
        tokenizer, 
        seed,
        pool_hs_type, # pool_hs_type='single_last',
    )
    model = embed_initializer.initializeEmbed(
        model, 
        tokenizer, 
        train_token2tokenid, 
        init_vec_type, 
        category_to_concepts_for_vec,
        category2initoken_ids,
        layer_idx,
        print_flag=print_flag
    )

    criteria = torch.nn.CrossEntropyLoss()
    model.train()
    return model, criteria




# ********************* train **********************

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
            
        # 各(架空)概念毎に1sample表示する
        print(f"Example train 1 sample for concept '{target_concept}':")
        print(train_samples[-1])
        print("**********")

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


def train(model_size, 
          model, 
          tokenizer, 
          criteria, 
          concept_to_train_data_source, train_sample_format, conceptForFict2token_map,  # train_samples, 
          memTokenIds, 
          padTokenId, 
          save_mem_dir, 
          lr=0.01, 
          maxEpochs=50, 
          earlyStoppingCount=5
    ):
    """
    [WIP] earlyStoppingCount は未使用になっている
    """
    if int(model_size) == 1:
        # global BATCH_SIZE
        BATCH_SIZE = 256 # 512
        factor=0.5
        min_lr=1e-04
        if maxEpochs >= 300:
            patience=100 # 300
            cooldown=200 # 100
        else:
            # maxEpochsが小さい場合はpatience, cooldownも小さくする
            patience=30
            cooldown=20
    
    elif int(model_size) == 2:
        BATCH_SIZE = 128 #256
        factor=0.9
        min_lr=1e-07
        if maxEpochs >= 300:
            patience=50
            cooldown=150
        else:
            # maxEpochsが小さい場合はpatience, cooldownも小さくする
            patience=30
            cooldown=10

    elif int(model_size) == 4:
        BATCH_SIZE = 4 # 16 (zao01なら16で行けそう) 4 (zao00は8でout of memoryになった)
        factor=0.9
        min_lr=1e-07
        if maxEpochs >= 300:
            patience=50
            cooldown=150
        else:
            # maxEpochsが小さい場合はpatience, cooldownも小さくする
            patience=30
            cooldown=10
    
    elif int(model_size) == 9:
        BATCH_SIZE = 32 # 64
        factor=0.95 #0.5
        min_lr=5e-08  #1e-06
        patience=30 # 100
        cooldown=100 # 500

    elif int(model_size) == 12:
        BATCH_SIZE = 4 #8 #4 # 16 # 64
        factor=0.95 #0.5
        min_lr=5e-08  #1e-06
        patience=30 # 100
        cooldown=100 # 500
    else:
        pass


    params = {
        'lr':lr, 
        'betas':(0.9, 0.95), 
        'weight_decay':0.0
    }
    try:
        opt = torch.optim.AdamW(
            model.model.embed_tokens.parameters(),
            **params
        )
    except:
        opt = torch.optim.AdamW(
            model.model.language_model.embed_tokens.parameters(),
            **params
        )

    scheduler = ReduceLROnPlateau(
        opt,
        mode='min',        # 'min': 減る=改善 (例: loss), 'max': 増える=改善 (例: accuracy)
        factor=factor,        # 悪化が続いたら lrを*倍に小さくする. 0.6B:0.5, 4B:0.8
        patience=patience,        # 何エポック分の非改善を許すか. bestの値(最も低いloss)から数えてこのエポック数分でthreshold以上の大きさの改善が見られなければlrを下げるか.
        threshold=1e-3,    # 「改善」とみなす最小変化量. threshold_mode='rel' の場合は相対的な変化量つまり%, 'abs' の場合は絶対的な変化量で判定
        threshold_mode='rel',  # 'rel' は相対, 'abs' は絶対差で判定
        cooldown=cooldown,        # lr 低下後、何エポックは様子見するか
        min_lr=min_lr, # 0.0,        # 下限
        eps=0, # 1e-05,         # lrの変化が微小すぎるときの更新抑制. old_lr - new_lr <= eps なら更新しない
    )

    accLog = {}
    print("Start training...")

    for epoch in tqdm(range(maxEpochs)):
        print('Epoch %d/%d'%(epoch+1, maxEpochs))

        # *** epoch毎にfact_sentencesの組み合わせをシャッフルしてデータを構成し直す ***
        train_samples = constructTrainSamples(concept_to_train_data_source, train_sample_format, conceptForFict2token_map, n_feat_in_a_sample)
        trainingData, evalInputs, evalOutputTexts, indices = encodeTrainSamplesWithTokenizer(train_samples, tokenizer, padTokenId, model.device)

        # このepochの学習
        model.train()
        random.shuffle(indices)
        totalLoss = 0.0

        for step, i in enumerate(range(0, len(indices), BATCH_SIZE)):
            batchIds = indices[i:i+BATCH_SIZE]
            input_ids = trainingData[batchIds]
            attention_mask = (input_ids != padTokenId).long() # e.g. [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]] のような形で、padTokenIdの位置が0になるマスクを作成
            token_type_ids = torch.zeros_like(input_ids)    # 2つのシーケンスを識別するバイナリマスク. e.g. [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] のような形 (全て0でOK)
            
            labels = input_ids.clone()
            # labels[labels == padTokenId] = -100 # PAD の位置は loss 計算から無視させたい。CrossEntropyLossのignore_indexに合わせて、padTokenIdの位置を-100に変換しておく
            labels[attention_mask == 0] = -100    # pad_token_id == eos_token_id だと eosも-100になってしまうため、attention_maskが0の位置、つまりpadTokenIdの位置を-100に変換しておく

            opt.zero_grad()
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = output.loss
            loss.backward()
            opt.step()

            # 勾配の確認
            if epoch == 0:
                try:
                    if sum(model.model.embed_tokens.weight.grad[1000]) == 0:
                        print(f"OK: 学習対象でないtokenID {1000} の勾配が0です")
                    if sum(model.model.embed_tokens.weight.grad[memTokenIds[0]]) != 0:
                        print(f"OK: 学習対象tokenID {memTokenIds[0]} の勾配が0ではありません")
                except:
                    if sum(model.model.language_model.embed_tokens.weight.grad[1000]) == 0:
                        print(f"OK: 学習対象でないtokenID {1000} の勾配が0です")
                    if sum(model.model.language_model.embed_tokens.weight.grad[memTokenIds[0]]) != 0:
                        print(f"OK: 学習対象tokenID {memTokenIds[0]} の勾配が0ではありません")
            # model.model.embed_tokens.weight[trainTokenIds] *= (1 - 0.01 * 0.1)  # lr * wd  # weight-decayを使う場合はここで手動
            

            # WandB logging
            log_interval=1 # batch_sizeを大きくしているので小さめ

            # log_intervalが1epoch内のstep数よりも大きい値に設定されている場合は1epoch毎にログを出力
            if log_interval > len(batchIds):
                log_interval = len(batchIds)
            
            if step % log_interval == 0:
                # print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  # MB単位
                gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                wandb.log({
                    "loss": loss.item(), 
                    "epoch": epoch, 
                    "step": step, 
                    # "param_sum": next(model.parameters()).data.sum().item(),
                    "lr": opt.param_groups[0]["lr"], 
                    "gpu_memory_allocated_MB": gpu_memory,
                    "gpu_memory_reserved_MB": gpu_reserved
                })
            totalLoss += loss.item()
            # totalLoss += loss.detach().cpu().tolist()


        # 検証データはないため，train lossでschedulerを更新する. 余裕があれば検証データ（別ルールによるtripletの言い換え)を作り，それをevaluateModelに入れてlossを計算するようにする
        num_steps = math.ceil(len(indices) / BATCH_SIZE)
        avgLoss = totalLoss / num_steps
        if scheduler is not None:
            scheduler.step(avgLoss)
        print('epoch %d loss:' % (epoch+1), avgLoss)



        # **** 検証とschedulerの更新 ****
        if epoch%10 == 0:
            acc = evaluateModel(model, tokenizer, evalInputs, evalOutputTexts, verbose=True)
            accLog[epoch] = {'eval acc': acc}
            print(acc)
            if acc==1.0:
                break

        if epoch%100 == 0 or epoch in list(range(10)) + [10, 15, 20, 25, 40, 50, 60, 70, 80, 90]:
            save_mem_path = os.path.join(save_mem_dir, f"{epoch}.pth")
            save_mem_vec(model, memTokenIds, save_mem_path)
            print(f"trained vecs at {epoch} are saved in {save_mem_path}.")
                        
    return model, accLog







# *************************************************************** main ***************************************************************
def main(args):
    print_flag = False

    seed = args.seed
    model_size = args.model_size
    lr = args.lr
    maxEpochs = args.max_epochs
    target_concepts_filename = args.target_concepts_filename
    init_vec_type = args.init_vec_type
    pool_hs_type = args.pool_hs_type
    layer_idx = args.layer_idx

    trained_date = datetime.now().strftime("%Y%m%d")


    # ** モデル保存dirnameの設定 **
    global model_name_for_dirname
    # [WIP] 'it'と'pt'のどちらが良いかは未検証.とりあえず'it'で統一.
    if model_size in ['2', '9']:
        model_version = 2
    elif model_size in ['1', '4', '12']:
        model_version = 3
    else:
        pass

    model_name = f"google/gemma-{model_version}-{model_size}b-it" # [memo] 'gemma-'部分は変えないこと!! -を消すとモデルがloadできない．さらにそのエラーメッセージは，"huggingface-cli login"をして，という関係ないmessageになるので注意!
    model_name_for_dirname = f"gemma-{model_version}-{model_size}B-lr{lr}-{trained_date}"
    if layer_idx is not None:
        model_name_for_dirname += f"-hidden_layer{layer_idx}"
    model_name_for_dirname += f"-seed{seed}"


    # *** tokenizer/modelをload. ただしmodelの設定はここではまだ行わない ***
    tokenizer = AutoTokenizer.from_pretrained(model_name) #, token=access_token)

    if 'gemma-' not in model_name.lower():
        print('The specified model does not seem Gemma3-based model.')
        print('Calculation for non-Gemma3 models is not implemented yet [TODO]')
        raise ValueError("The specified model does not seem Gemma3-based model.")
    
    # model = None
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")#, token=access_token)

    if tokenizer.pad_token_id is None:
        # llama系の場合はpad_tokenが設定されていないことがあるため，以下のようにeos_tokenをpad_tokenに設定する. gemma3は設定済みだった
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id


    # *** dir/file path 設定 ***
    train_data_dir = os.path.join(project_root, 'data', 'train_data') # 🟠train_data_dir = os.path.join(project_root, 'data', 'triplets')

    # *** 🟠 修正後(2026/01/18~) 引数でconfig/{target_concepts_filename}で指定されたconcept群を学習対象とするように変更 ***
    class_to_target_concepts_path = os.path.join(project_root, 'config', target_concepts_filename)
    print(f"Loading target concepts from: {class_to_target_concepts_path}")
    if not os.path.exists(class_to_target_concepts_path) or target_concepts_filename.split('.')[-1] != 'json':
        raise ValueError(f"指定されたtarget_concepts_filename '{target_concepts_filename}' が存在しないか，jsonファイルではありません。configディレクトリ内の正しいjsonファイル名を指定してください。")
    with open(class_to_target_concepts_path, 'r') as f:
        class_to_target_concepts_config = json.load(f)
    config_concept_list = sum(class_to_target_concepts_config.values(), [])
    print(f"Target concepts specified in config: {config_concept_list}")
    save_mem_dir = os.path.join(project_root, "memvec_models", f"{model_name_for_dirname}_{target_concepts_filename.replace('.json', '')}_initvecwith{init_vec_type.replace(' ', '_')}")

    # もしすでに同名のディレクトリが存在していたら、_2のように末尾に連番をつける
    original_save_mem_dir = save_mem_dir
    counter = 2
    while os.path.exists(save_mem_dir):
        save_mem_dir = f"{original_save_mem_dir}_{counter}"
        counter += 1
    print('save_mem_dir:', save_mem_dir)

    os.makedirs(save_mem_dir, exist_ok=True)



    # ********* train data取得準備 *********
    # filter: 全てのカテゴリ・固有名詞リスト の辞書を読み込む (重複等のfiltering済み)
    category_properNouns_dict = loadProperNounData(
        propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept,
        print_flag=print_flag
    )

    # filter: [memo] もう新しいwikiページを読み込みたくない場合は、すでに保存済みのwikiページがあるpropernounのみにフィルタリングする
    if dont_get_new_wiki_flag:
        print("Filtering proper nouns to those with already saved wiki pages...")
        filtered_category_properNouns_dict = {}
        for category, propernouns in category_properNouns_dict.items():
            filtered_category_properNouns_dict[category] = filterProperNounsWithWikiPage(propernouns, wiki_page_save_dir)
        print(f"concept num: {sum(len(concepts) for concepts in category_properNouns_dict.values())} \
              -> {sum(len(concepts) for concepts in filtered_category_properNouns_dict.values())}")
    else:
        filtered_category_properNouns_dict = category_properNouns_dict
    
    # filter: concept数が propnoun_num_for_init_vec + propnoun_num_for_new_concept 程度 以上のカテゴリのみを残す
    keep_category_list = []
    for category, concepts in filtered_category_properNouns_dict.items():
        if len(concepts) >= propnoun_num_for_init_vec + propnoun_num_for_new_concept - 5:
            keep_category_list.append(category)
    filtered_category_properNouns_dict = {category: filtered_category_properNouns_dict[category] for category in keep_category_list}
    print(f"After filtering categories with enough proper nouns, category num: {len(filtered_category_properNouns_dict)}, concept num: {sum(len(concepts) for concepts in filtered_category_properNouns_dict.values())}")


    # ***** target_concept_listに学習対象のconcept名を追加 *****
    # ** 学習データが存在するconcept名のみを抽出.
    trainable_concept_list = []
    for filename in os.listdir(train_data_dir):
        concept_name = filename.split('.json')[0].replace('_', ' ')
        trainable_concept_list.append(concept_name)
    print(f"\ntrainable concepts: {trainable_concept_list}")

    # ** configファイルの指定に基づき，学習対象conceptを絞り込む
    if config_concept_list[0] not in [None, 'None']:
        # * 学習対象conceptが個別に指定されている場合. (config/target_concepts.jsonで指定):
        category_to_conceptsForFict = defaultdict(list)
        # config_concept_listに含まれ、学習データが存在し、所属するカテゴリが特定できるconceptのみを抽出
        for tcat, tconc_lst in class_to_target_concepts_config.items(): # for tc in config_concept_list:
            for tconc in tconc_lst:
                if tconc in trainable_concept_list:
                    category_to_conceptsForFict[tcat].append(tconc)
                else:
                    print(f"Warning: Concept '{tconc}' specified in config is not included in trainable_concept_list and will be skipped.")
    else:
        # * 学習対象conceptが個別に指定されていない場合: そのままtarget_concept_listの(学習可能な概念)全てを学習対象とする
        category_to_conceptsForFict = defaultdict(list)
        for category, concepts in filtered_category_properNouns_dict.items():
            for concept in concepts:
                if concept in trainable_concept_list:
                    category_to_conceptsForFict[category].append(concept)

    print("Target category - concept mapping: ", category_to_conceptsForFict)

    target_concept_list = sum(category_to_conceptsForFict.values(), [])
    target_concept_list = sorted(target_concept_list) # target_concept_list をアルファベット順にsort
    print(f"Class - target concepts mapping (after filtering with trainable concepts):")
    if print_flag:
        for category, concepts in category_to_conceptsForFict.items():
            print(f"  {category}: {concepts}")
    print("Target concept list:", target_concept_list, '\n')

    # 架空の概念用の固有名詞から、その所属カテゴリを引けるようにするためのmap
    conceptForFict2category_map = {conceptForFict: category for category, concepts in category_to_conceptsForFict.items() for conceptForFict in concepts} 
    print(f"categories in conceptForFict2category_map: {set(conceptForFict2category_map.values())} ...")

    
    # filter: あまりたくさんのデータを保持しておくとメモリが足りなくなるため、filtered_category_properNouns_dictには、訓練対象のカテゴリ+それ以外のカテゴリ10個をランダムに選んで保持する。この10個は他カテゴリ初期化時に利用する
    n = 5
    keep_category_list = list(category_to_conceptsForFict.keys()) + \
        random.sample(list(set(filtered_category_properNouns_dict.keys()) - set(category_to_conceptsForFict.keys())), \
                      min(n, len(filtered_category_properNouns_dict)-len(category_to_conceptsForFict)))
    filtered_category_properNouns_dict = {category: filtered_category_properNouns_dict[category] for category in keep_category_list}
    print(f"After filtering categories to target categories + {n} random categories, category num: {len(filtered_category_properNouns_dict)}, concept num: {sum(len(concepts) for concepts in filtered_category_properNouns_dict.values())}")


    # ***** 各カテゴリ内で、vec初期化に使用する固有名詞を取得 *****
    # # 各概念毎に、架空の概念用の特徴の生成に成功した(=架空の概念用の)固有名詞を取得する -> train data 作成時点で特徴獲得には成功している
    # category_to_concepts_for_fictconcept = loadConceptsForFictConcept()
    # 次に、各カテゴリの全固有名詞から、架空の概念用の特徴の生成に成功した固有名詞を除外し、残った固有名詞全てをvec初期化用の概念とする
    category_to_concepts_for_vec = {}
    for category in filtered_category_properNouns_dict.keys():
        propernouns_for_init_vec = list(set(filtered_category_properNouns_dict[category]) - set(category_to_conceptsForFict.get(category, [])))
        if len(propernouns_for_init_vec) < 1:
            # もし、そもそも固有名詞が1つもないカテゴリがあれば、そのカテゴリはcategory_to_concepts_for_vecに含めない
            continue
        category_to_concepts_for_vec[category] = propernouns_for_init_vec
        if print_flag:
            print(f"category: {category}, proper nouns for vec initialization: {len(propernouns_for_init_vec)}, {propernouns_for_init_vec[:5]} ...") # 先頭5個を表示


    # ********* train data取得準備 *********
    # *** concept数の分だけ空きtokenを確保する ***
    trainTokenIds = [tokenizer.convert_tokens_to_ids(f'<unused{i}>') for i in range(len(target_concept_list))]
    trainTokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in trainTokenIds]

    train_token2tokenid = {}
    for id, token in zip(trainTokenIds, trainTokens):
        train_token2tokenid[token] = id
    if len(trainTokens) > 6: # 先頭3つと末尾3つを表示するための条件分岐
        print(f"train target tokens: {len(trainTokens)}, {trainTokens[:3]} ... {trainTokens[-3:]}") # 先頭3つと末尾3つを表示
    else:
        print(f"train target tokens: {len(trainTokens)}, {trainTokens}") # 全て表示


    # *** (架空の)concept名: 空きトークン の割り当て辞書作成 ***
    conceptForFict2token_map = {}
    memTokenIds = []
    for target_concept, trainable_token, token_id in zip(target_concept_list, trainTokens, trainTokenIds):
        conceptForFict2token_map[target_concept] = trainable_token
        memTokenIds.append(token_id)
        # print(f"{token_id}: {target_concept} -> {trainable_token}")

    # concept-token割り当てを保存
    save_path = os.path.join(save_mem_dir, "token_assignment.json")
    with open(save_path, "w") as f:
        json.dump(conceptForFict2token_map, f, ensure_ascii=False, indent=4)
    print(f"Saved conceptForFict2token_map to {save_path}")


    # *** categoryごとに、割り当てた空きtoken idをリストにまとめる. ***
    # これは、'category_COG' 系の初期化の場合、categoryごとのvec初期化の際に、同じカテゴリの概念に割り当てたtokenのidをまとめて初期化するために使用する. 
    # (カテゴリ毎に重心vecを作成するため、同じカテゴリ内の概念に該当する空きtokenは同じ重心vecで初期化するから。)
    category2initoken_ids = defaultdict(list)   # category -> [unused_token_id1, unused_token_id2, ...]
    for conceptForFict in conceptForFict2token_map.keys():
        category = conceptForFict2category_map.get(conceptForFict)
        if category is None:
            raise ValueError(f"Concept '{conceptForFict}' not found in any category.")
        tk = conceptForFict2token_map[conceptForFict]
        tk_id = train_token2tokenid[tk]
        category2initoken_ids[category].append(tk_id)



    # ****** tripletを読み込み学習データを構築 ******
    concept_to_train_data_source = {}
    for target_concept in target_concept_list:
        concept_to_train_data_source[target_concept] = []

        # 対応する空token名を取得. <unused0>など
        unused_token = conceptForFict2token_map[target_concept]

        # load data
        filename = target_concept.replace(' ', '_') + '.json'
        with open(os.path.join(train_data_dir, filename), 'r') as f:
            data = json.load(f)
        wiki_text = data['summary'] # data['text']も選べるが、fact sentencesに比べて大き過ぎるのでsummaryを使用している
        facts = data['facts']

        # concept名/"It" を割り当てられた空tokenに置換 (大小区別なし)
        wiki_text_with_token = re.sub(re.escape(target_concept), unused_token, wiki_text, flags=re.IGNORECASE)
        facts_with_token = [fact.replace('It', unused_token) for fact in facts]

        concept_to_train_data_source[target_concept].append({'wiki_text_with_token': wiki_text_with_token, 'facts_with_token': facts_with_token})

    with open(os.path.join(project_root, 'data', 'templates', 'train_sample_format.json'), "r") as f:
        train_sample_format = json.load(f)


    # # Wandb設定
    PROJECT_NAME = os.path.basename(save_mem_dir) # save_mem_dirの一番最後だけ取ってくる
    wandb_dir = os.path.join(project_root, "memvec_wandb_logs", model_name_for_dirname)
    if model is not None:
        set_wandb_env(PROJECT_NAME, model_name_for_dirname, wandb_dir, WANDB_API_KEY)


    # ********* model等の準備: 予約済み特殊トークンの埋め込みの初期化など *********
    model, criteria = prepareGemmaModel(
        model_name,
        save_mem_dir,
        model, 
        tokenizer,
        train_token2tokenid, 
        init_vec_type, 
        pool_hs_type,
        category_to_concepts_for_vec,
        category2initoken_ids,
        seed,
        layer_idx,
        print_flag=print_flag
    )
    padTokenId = tokenizer.vocab[tokenizer.pad_token]
    



    # ********* train *********
    start_time = time.time()
    model, accLog = train(
        model_size,
        model,
        tokenizer,
        criteria, 
        concept_to_train_data_source, train_sample_format, conceptForFict2token_map, # train_samples,
        memTokenIds,
        padTokenId, 
        save_mem_dir,
        lr, 
        maxEpochs, 
        earlyStoppingCount=5
    )
    print('accLog:', accLog)

    # detailedHistory = {epoch:{c:accLog[epoch]['detailed'][c] for c in accLog[epoch]['detailed']} for epoch in accLog}
    detailedHistory = accLog
    df = pd.DataFrame(detailedHistory)
    # df.to_csv(args.output)

    # *** save ***
    save_hist_path = f'{project_root}/memvec_training_history/{model_name_for_dirname}.csv'
    os.makedirs(os.path.dirname(save_hist_path), exist_ok=True)
    df.to_csv(save_hist_path)

    # 最後のmemory vectorをセーブしたい場合
    save_mem_path = os.path.join(save_mem_dir, f"{maxEpochs}.pth")
    save_mem_vec(model, memTokenIds, save_mem_path)
    print(f"trained vecs at {maxEpochs} are saved in {save_mem_path}.")


    # 訓練ループ全体の時間を計測
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    wandb.finish()










# ********************* 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_concepts_filename', type=str, default='target_concepts.json', help='学習対象とするconcept群を指定したjsonファイル名 (configディレクトリ内). 例: "target_concepts.json"')
    parser.add_argument('--model_size', type=str, default='12', help='モデルサイズ (例: 4, 9, 12)')
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    parser.add_argument('--max_epochs', type=int, default=600, help='最大エポック数')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='CUDA_VISIBLE_DEVICESの設定. ただし数字は1つだけ指定すること. 例: "2"')
    parser.add_argument('--init_vec_types', type=str, nargs='+', default=['zero', 'uniform', 'norm_rand'], help='memory vectorの初期化方法のリスト. ')
    parser.add_argument("--pool_hs_type", type=str, default="eos", choices=["eos", "last_token", "mean_pool"], help="隠れ状態のプーリング方法。")
    parser.add_argument('--layer_indices', type=int, nargs='*', default=None, help='隠れ状態を取得する層のインデックス。-1なら最終層、0以上の整数ならその層の隠れ状態を使用する。init_vec_typeが \'category_centroid_by_hidden_state_mean\' の場合に使用')
    parser.add_argument('--thread_id', type=int, nargs='?', default=0, help='複数process同時に実行する場合のthread id (0 or 1). これにより,実行する設定(seed, init_vec_typeの組)が被らないように調整する')
    parser.add_argument('--process_num', type=int, nargs='?', default=2, help='同時に実行するprocess数')
    parser.add_argument('--seed_num', type=int, nargs='?', default=10, help='シードの数. 例えば10に設定した場合、seed0からseed9までの10個のシードで学習を実行することになる。')
    args = parser.parse_args()

    processNum = args.process_num # 同時に実行するprocess数
    print(f"\t layer_indices: {args.layer_indices}")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        
    task_id = -1
    for seed in range(args.seed_num):
        args.seed = seed # mainにargsとして渡すためにargs.seedに代入している。main内でargs.seedを参照することで、現在のシード値を取得できるようになる。
        
        init_vec_type_lst = args.init_vec_types

        for init_vec_type in init_vec_type_lst:
            
            layer_indices = args.layer_indices # ここでinit_vec_typeループ毎に読み込まないと、layer_indices = [None] が代入されたループの次のループでも[None]のままになってしまう

            if len(layer_indices) < 1: #  or 'HS' not in init_vec_type:  
                # layer_idxが不要の初期化方法の場合は、layer_indicesを[None]にして、1回だけループするようにする
                print(f"init_vec_type: {init_vec_type}, layer_indices: {layer_indices}")
                layer_indices = [None]

                
            for layer_idx in layer_indices:
                args.layer_idx = layer_idx


                print(f"\n\n=== Training with seed: {seed}, init_vec_type: {init_vec_type}, layer_idx: {layer_idx} ===")
                args.init_vec_type = str(init_vec_type)

                task_id += 1

                if task_id % processNum != args.thread_id:
                    # 複数process同時に実行する場合, thread_idに応じてtask_idが偶数or奇数の設定のみを実行する
                    print(f"Skipping task_id {task_id} for thread_id {args.thread_id}")
                    continue

                fix_seed(seed)
                main(args)

                # GPUメモリ解放
                torch.cuda.empty_cache()
                # 3秒待機
                time.sleep(3)