from processing import *
from similarity import *
from visualize import *

from rank_bm25 import BM25Okapi
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import torch
import os
import time
import numpy as np

models = {}
DEVICE = None

# --- get concatenated text ---
def getConcText(dict, key_list,isShowLen=False):
    conc_text = ""
    for attr in key_list:
        if attr in dict.keys():
            conc_text = conc_text + str(dict[attr])
    if isShowLen:
        print(f'\n\nconc_text： {len(conc_text)}\n\n')
    return conc_text

# --- Exp setup ---
isAddSKR, isAddCrime, isAddFull, \
isFullText, isFourTier, isFourTierOnly, \
isSelectModel \
\
= False, False, False, \
  False, False, False, \
  False

# isAddSKR = True
# isAddCrime = True
# isAddFull = True
# isFullText = True
# isFourTier = True
isFourTierOnly = True

# --- set parameters ---
global signal
signal = ""
key_q_list = ['q']
key_c_list = ['ajjbqk']

if isAddSKR:
    print(f'\nisAddSCR: {isAddSKR}\n')
    signal = 'SKR'
    key_q_list = ['crime', 'q', 'des']
    key_c_list = ['crime', 'ajjbqk']
elif isAddCrime: #Simplified
    print(f'\nisAddCrime: {isAddCrime}\n')
    signal = 'Simplified'
    key_q_list = ['q', 'crime']
    key_c_list = ['ajjbqk', 'crime']
elif isAddFull:
    print(f'\nisAddFull: {isAddFull}\n')
    signal = 'Full'
    key_q_list = ['crime', 'q', 'des']
    key_c_list = ['crime', 'ajjbqk','cpfxgc','pjjg']
elif isFullText:
    print(f'\nisFullText: {isFullText}\n')
    signal = 'SimpleFullText'
    key_q_list = ['crime', 'q', 'des']
    key_c_list = ['qw']
elif isFourTier:
    print(f'\nisFourTier: {isFourTier}\n')
    signal = 'FourTier'
    key_q_list = ['crime', 'q', 'des']
    key_c_list = ['crime', 'ajjbqk', 'des']
elif isFourTierOnly:
    print(f'\nisFourTierOnly: {isFourTierOnly}\n')
    signal = 'FourTierOnly'
    key_q_list = ['crime', 'des']
    key_c_list = ['crime', 'des']
else:
    pass

# --- similarity accelerated ---

def setup_device():
    """set GPU"""
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # set the GPU storage management strategy
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device('cpu')
        print("Using CPU")
    return DEVICE


def load_model_from_local(model_name, model_type):
    global DEVICE
    input_model_path = os.path.join(os.getcwd(), 'models', model_name)
    print(f"input_model_path: {input_model_path}")

    if not os.path.exists(input_model_path):
        print(f"Errors | input_model_path NOT exists: {input_model_path}")
        print("Please download model to correct folder:")
        print(f"   {os.getcwd()}/models/{model_name}/")
        return None, None

    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"load_model_from_local {model_type.upper()} MODEL: {input_model_path}")

        required_files = ['config.json', 'pytorch_model.bin']
        for file in required_files:
            if not os.path.exists(os.path.join(input_model_path, file)):
                print(f"Errors | missing necessary files {file}")
                return None, None

        # Tokenizer
        if model_type == "longformer":
            tokenizer_files = ['tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
            missing_files = []
            for file in tokenizer_files:
                if not os.path.exists(os.path.join(input_model_path, file)):
                    missing_files.append(file)

            if missing_files:
                print(f"Tokenizer files missing: {missing_files}")
                try:
                    # online access trying
                    from huggingface_hub import snapshot_download
                    print("Downloading tokenizer files...")
                    snapshot_download(
                        repo_id="allenai/longformer-base-4096",
                        allow_patterns=["tokenizer*", "vocab.*", "special_tokens*"],
                        local_dir=input_model_path
                    )
                    print("√ tokenizer downloaded.")
                except Exception as e:
                    print(f"× Cannot download tokenizer: {e}")
                    return None, None

        # AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)

        # loading model
        model = AutoModel.from_pretrained(
            input_model_path,
            torch_dtype=torch.float16,  # decrease the memory usage
            low_cpu_mem_usage=True
        )

        model = model.to(DEVICE)
        model.eval()

        print(f"√ Local Model accesssed: {model_type.upper()}")
        print(f"model.dtype: {model.dtype}")

        return tokenizer, model

    except Exception as e:
        print(f"× Cannot load local model: {e}")
        import traceback
        traceback.print_exc()

        # try create tokenizer
        if model_type == "longformer":
            print("try to load Longformer...")
            try:
                from transformers import LongformerTokenizer, LongformerModel
                tokenizer = LongformerTokenizer.from_pretrained(input_model_path)
                model = LongformerModel.from_pretrained(input_model_path)
                model = model.to(DEVICE)
                model.eval()
                print("√ Longformer load successfully.")
                return tokenizer, model
            except Exception as e2:
                print(f"× Unable to load: {e2}")

        return None, None


def load_model(model_type="bert"):
    global models, DEVICE, isKaggle

    if DEVICE is None:
        setup_device()

    if model_type in models:
        return models[model_type]

    print(f"Loading model: {model_type.upper()} ...")
    from transformers import AutoModel, AutoTokenizer

    start_time = time.time()

    try:
        if isKaggle:
            print("isKaggle： accessing online models")
            if model_type == "bert":
                model_name = "bert-base-chinese"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
            elif model_type == "longformer":
                try:
                    model_name = "schen/longformer-chinese-base-4096"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                except:
                    model_name = "allenai/longformer-base-4096"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
            elif model_type == "roberta":
                model_name = "hfl/chinese-roberta-wwm-ext"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
            elif model_type == "lawformer":
                model_name = "lawformer"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported model: {model_type}")
        else:
            # 非Kaggle环境：从本地加载
            print("Local environment: accessing local models.")

            # 定义模型名称到本地目录名的映射 | Define the mapping route from model name to local path
            model_mapping = {
                "bert": "bert-base-chinese",
                "longformer": "longformer-base-4096",
                "roberta": "chinese-roberta-wwm-ext",
                "lawformer": "lawformer"
            }

            if model_type not in model_mapping:
                raise ValueError(f"Unsupported model_type: {model_type}")

            model_name = model_mapping[model_type]

            # 检查模型是否存在 | Check if the model exists
            model_path = os.path.join(os.getcwd(), 'models', model_name)
            if not os.path.exists(model_path):
                print(f"× Error | Not exist model_type {model_type.upper()}, please check the local model path.")
                print(f"! Download the model to model_path： {model_path}")
                exit(1)

            # 从本地加载模型 | Load local model
            tokenizer, model = load_model_from_local(model_name, model_type)
            if tokenizer is None or model is None:
                print(f"× Failed to load the local model.")
                exit(1)

        # 将模型移动到设备 | Move the model to devive
        model = model.to(DEVICE)
        model.eval()

        models[model_type] = {
            'tokenizer': tokenizer,
            'model': model,
            'name': model_type
        }

        load_time = time.time() - start_time
        print(f"√ Model_type load successfully {model_type.upper()}. Time: {load_time:.2f} seconds.")

        return models[model_type]

    except Exception as e:
        print(f"× Failed to load the local model. {e}")
        import traceback
        traceback.print_exc()
        if not isKaggle:
            print("! Please check the local model path.")
        exit(1)


def get_embedding(text, model_type="bert"):
    """获取文本的嵌入向量"""
    global models

    if model_type not in models:
        load_model(model_type)

    model_info = models[model_type]
    tokenizer = model_info['tokenizer']
    model = model_info['model']

    try: # memory restrictions
        if model_type == "bert":
            max_length = 256
        elif model_type == "longformer":
            max_length = 512
        elif model_type == "roberta":
            max_length = 256
        elif model_type == "lawformer":
            max_length = 256

        # clear GPU temporary memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 使用平均池化获取句子嵌入 | mean average embeeding
            embeddings = outputs.last_hidden_state.mean(dim=1)

        embedding_vector = embeddings.cpu().numpy().flatten()

        # 释放GPU内存 | free GPU memory
        del inputs, outputs, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 确保维度一致 | ensure consistent dimension
        if embedding_vector.shape[0] != 768:
            if embedding_vector.shape[0] > 768:
                embedding_vector = embedding_vector[:768]
            else:
                embedding_vector = np.pad(embedding_vector, (0, 768 - embedding_vector.shape[0]))

        return embedding_vector

    except Exception as e:
        print(f"{model_type.upper()} error | Embedding calclulation error: {e}")
        # 清理GPU缓存 | clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.zeros(768)

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    try:
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()

        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    except Exception as e:
        print(f"Error | cosine_similarity: {e}")
        return 0.0


def sim_multimodel(text1, text2, model_type="bert"):
    """多模型相似度计算"""
    try:
        embedding1 = get_embedding(text1, model_type)
        embedding2 = get_embedding(text2, model_type)

        similarity = cosine_similarity(embedding1, embedding2)
        return similarity

    except Exception as e:
        print(f"{model_type.upper()} Error | failed to calculate similarity: {e}")
        return 0.0


def sim_bert(text1, text2):
    return sim_multimodel(text1, text2, "bert")


def sim_longformer(text1, text2):
    return sim_multimodel(text1, text2, "longformer")


def sim_roberta(text1, text2):
    return sim_multimodel(text1, text2, "roberta")


def sim_lawformer(text1, text2):
    return sim_multimodel(text1, text2, "lawformer")


def getCandidatesSim_multimodel(sim_model, ridx):
    if sim_model in [sim_bm25, sim_tfidf]:
        return sim_model(ridx)

    all_cids = get_all_cid(ridx)
    sim_dict = {}

    model_type = "bert"  #
    if sim_model.__name__ == 'sim_longformer':
        model_type = "longformer"
    elif sim_model.__name__ == 'sim_roberta':
        model_type = "roberta"
    elif sim_model.__name__ == 'sim_lawformer':
        model_type = "lawformer"

    print(f"! Using {model_type.upper()} to process ridx={ridx}, with {len(all_cids)} candidates in total.")

    # load model
    if sim_model.__name__ in ['sim_bert', 'sim_longformer', 'sim_roberta', 'sim_lawformer']:
        load_model(model_type)

    # 获取query文本和嵌入 | get query text and embeddings
    qt= getQueryDict(ridx)
    qt_text = getConcText(qt, key_q_list)
    query_text = getCorpus(qt_text)
    query_embedding = None

    if sim_model.__name__ in ['sim_bert', 'sim_longformer', 'sim_roberta', 'sim_lawformer']:
        print(f"📝 处理query文本，长度: {len(query_text)}")
        query_embedding = get_embedding(query_text, model_type)
        print(f"✅ Query嵌入维度: {query_embedding.shape}")

    # process each candidate
    for i, cid in enumerate(all_cids):
        try:
            candidate_dict = getCandidateDict(ridx, cid)
            c_text = getConcText(candidate_dict, key_c_list)
            corpus2 = getCorpus(c_text)
            print(f"cid: {cid}, len(corpus2): {len(corpus2)}")

            if sim_model.__name__ in ['sim_bert', 'sim_longformer', 'sim_roberta', 'sim_lawformer']:
                candidate_embedding = get_embedding(corpus2, model_type)
                similarity = cosine_similarity(query_embedding, candidate_embedding)

                # clear GPU memory
                del candidate_embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                similarity = sim_model(query_text, corpus2)

            sim_dict[cid] = similarity

            # if (i + 1) % 5 == 0 or (i + 1) == len(all_cids):
            #     print(f"📊 已处理 {i + 1}/{len(all_cids)} 个candidates")

        except Exception as e:
            print(f"Errors when calculating (cid={cid}): {e}")
            sim_dict[cid] = 0.0

    # order and return the results
    sorted_cids = [cid for cid, _ in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)]
    return {ridx: sorted_cids}


def getAllPred_multimodel(sim_model):
    all_ridx = [q_dict['ridx'] for q_dict in all_query_dict]

    model_type = "bert"
    if sim_model.__name__ == 'sim_longformer':
        model_type = "longformer"
    elif sim_model.__name__ == 'sim_roberta':
        model_type = "roberta"
    elif sim_model.__name__ == 'sim_lawformer':
        model_type = "lawformer"

    print(f"Start to process {len(all_ridx)} quries， with model_type: {model_type.upper()}")

    all_pred = {}

    # load the model
    if sim_model.__name__ in ['sim_bert', 'sim_longformer', 'sim_roberta', 'sim_lawformer']:
        load_model(model_type)

    for i, ridx in enumerate(all_ridx):
        try:
            result = getCandidatesSim_multimodel(sim_model, ridx)
            all_pred.update(result)

            # clear memory when ridx processed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Errors when processing (ridx={ridx}): {e}")

    dump_json(all_pred, sim_model,signal)
    return all_pred


def run_specific_model(model_type="bert"):
    if model_type == "bert":
        sim_model, model_name = sim_bert, "BERT"
    elif model_type == "longformer":
        sim_model, model_name = sim_longformer, "Longformer"
    elif model_type == "roberta":
        sim_model, model_name = sim_roberta, "RoBERTa"
        # if not isKaggle and not os.path.exists(os.path.join(os.getcwd(), 'models', 'chinese-roberta-wwm-ext')):
        #     print("× Error | non-exist local model RoBERTa.")
        #     exit(1)
    elif model_type == "lawformer":
        sim_model, model_name = sim_lawformer, "Lawformer"
    else:
        print(f"× Error | not supported model_type: {model_type}")
        return

    print(f"Start to process with model_name: {model_name}")

    try: # Testing
        test_ridx = all_query_dict[0]['ridx'] if all_query_dict else 5156
        print(f"test_ridx: {test_ridx}")

        start_time = time.time()
        test_result = getCandidatesSim_multimodel(sim_model, test_ridx)
        test_time = time.time() - start_time

        print(f"√ Test successfully with: {test_time:.2f} seconds")
        print(f"√ Processed {len(test_result[test_ridx])} candidates")

        # Formal
        if test_time < 600:
            getAllPred_multimodel(sim_model)
            print("√ Processing down.")
        else:
            print("! Exceptional long processing time.")

    except Exception as e:
        print(f"× Runtime error: {e}")
        import traceback
        traceback.print_exc()


# === Accelerated ===
from rank_bm25 import BM25Okapi
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def process_single_cid(args):
    ridx, cid, sim_model = args
    print(f'{cid}\t', end="")
    corpus1 = getCorpus(getQueryDict(ridx)['q'])
    corpus2 = getCorpus(getCandidateDict(ridx, cid)['ajjbqk'])
    similarity = sim_model(corpus1, corpus2)
    return cid, similarity


def getCandidatesSim(sim_model, ridx): # multi-thread acceleraating
    '''for 1 ridx, get a ranked cid list using parallel processing'''
    if sim_model in [sim_bm25, sim_tfidf]:
        return sim_model(ridx)

    all_cids = get_all_cid(ridx)
    args_list = [(ridx, cid, sim_model) for cid in all_cids]
    sim_dict = {}
    max_workers = min(multiprocessing.cpu_count(), len(all_cids))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_cid, args): args for args in args_list}
        # pbar = tqdm(total=len(args_list), desc=f"processing candidates of ridx={ridx}", leave=False)

        # get results
        for future in as_completed(futures):
            try:
                cid, similarity = future.result()
                sim_dict[cid] = similarity
            except Exception as e:
                print(f"Error with processing (ridx={ridx}, cid={futures[future][1]}): {str(e)}.")

    # order and get results
    sorted_cids = [cid for cid, _ in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)]
    return {ridx: sorted_cids}


def process_single_ridx(args):
    sim_model, ridx = args
    return ridx, getCandidatesSim(sim_model, ridx)


def getAllPred(sim_model):
    '''for all ridx, collect all ranked cid list in the dict using parallel processing'''
    all_ridx = [q_dict['ridx'] for q_dict in all_query_dict]
    all_pred = {}

    args_list = [(sim_model, ridx) for ridx in all_ridx]

    max_workers = min(multiprocessing.cpu_count(), len(all_ridx))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_ridx, args): args for args in args_list}
        # pbar = tqdm(total=len(args_list), desc=f"all quries processing with ({sim_model.__name__})")

        # get results
        for future in as_completed(futures):
            try:
                ridx, result = future.result()
                all_pred[ridx] = result
            except Exception as e:
                print(f"Errors with processing (ridx={futures[future][1]}): {str(e)}.")

    dump_json(all_pred, sim_model,signal)
    return all_pred

def getCandidatesSim(sim_model, ridx):
    '''for 1 ridx, get a ranked cid list'''
    if sim_model == sim_bm25:
        return sim_model(ridx)
    elif sim_model == sim_tfidf:
        return sim_model(ridx)
    else:
        all_cids = get_all_cid(ridx)
        sim_dict = {}
        for cid in all_cids:
            corpus1 = getCorpus(getQueryDict(ridx)['q'])
            corpus2 = getCorpus(getCandidateDict(ridx,cid)['ajjbqk'])
            similarity = sim_model(corpus1, corpus2)
            sim_dict[cid] = similarity
        sorted_cids = [cid for cid, _ in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)]
        return {ridx: sorted_cids}

# sid = 1
# sim_models = [sim_jaccard,  sim_cosine, sim_tfidf, sim_bm25, sim_seq,  sim_COMP, sim_bert] #sim_wordvec,
# getCandidatesSim(sim_models[sid], ridx=5156)
# # sim_bm25(ridx)

def getAllPred(sim_model):
    '''for all ridx, collect all ranked cid list in the dict'''
    all_ridx = [q_dict['ridx'] for q_dict in all_query_dict]
    all_pred = {}
    for ridx in all_ridx:
        all_pred[ridx] = getCandidatesSim(sim_model, ridx)
    dump_json(all_pred,sim_model,signal)

def dump_json(results,sim_model,signal):
    # results.sort(key=lambda x: x[1], reverse=True)
    if not signal=='':
        signal = f'_{signal}'
    json_path = os.path.join(output_path, '1-Sim', f'{sim_model.__name__}{signal}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)
        print(f'dump json: {json_path}')

# TF-IDF, BM25 with multi-thread
if __name__ == "__main__":
    print(f'signal: {signal}')
    sid = 1 # select similarity function
    sim_models = [sim_tfidf, sim_bm25] # [sim_jaccard, sim_cosine, sim_tfidf, sim_bm25, sim_seq, sim_COMP]
    getAllPred(sim_models[sid-1])
    visualConfig(isVisualAll=False, pid=1)


# BERT
# if __name__ == "__main__":
#     setup_device()
#     run_specific_model("bert")      # BERT