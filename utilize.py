import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import random
from collections import defaultdict

from processing import *
from visualize import *

# isAddSKR, isAddCrime, isAddFull, \
# isFullText, isFourTier, isFourTierOnly, \
# isSelectModel \
# \
# = False, False, False, \
#   False, False, False, \
#   False
#
# # --- mode switch ---
# isSelectModel = True
#
# # isAddSKR = True
# # isAddCrime = True
# # isAddFull = True
# # isFullText = True
# isFourTier = True
# # isFourTierOnly = True
#
# # --- set parameters ---
# signal = "Retry"
# key_q_list = ['q']
# key_c_list = ['ajjbqk']
#
# # def setMode(isAddSKR=False,
# #             isAddCrime=False,
# #             isAddFull=False,
# #             isFullText=False,
# #             isFourTier=False,
# #             isFourTierOnly=False,
# #             isSelectModel=True
# #             ):
#
# if isAddSKR:
#     print(f'\nisAddSCR: {isAddSKR}\n')
#     signal = 'SKR_'
#     key_q_list = ['crime', 'q', 'des']
#     key_c_list = ['crime', 'ajjbqk']
# elif isAddCrime: #Simplified
#     print(f'\nisAddCrime: {isAddCrime}\n')
#     signal = 'Simplified_'
#     key_q_list = ['q', 'crime']
#     key_c_list = ['ajjbqk', 'crime']
# elif isAddFull:
#     print(f'\nisAddFull: {isAddFull}\n')
#     signal = 'Full_'
#     key_q_list = ['crime', 'q', 'des']
#     key_c_list = ['crime', 'ajjbqk','cpfxgc','pjjg']
# elif isFullText:
#     print(f'\nisFullText: {isFullText}\n')
#     signal = 'SimpleFullText_'
#     key_q_list = ['crime', 'q', 'des']
#     key_c_list = ['qw']
# elif isFourTier:
#     print(f'\nisFourTier: {isFourTier}\n')
#     signal = 'FourTier_'
#     key_q_list = ['crime', 'q', 'des']
#     key_c_list = ['crime', 'ajjbqk', 'des']
# elif isFourTierOnly:
#     print(f'\nisFourTierOnly: {isFourTierOnly}\n')
#     signal = 'FourTierOnly_'
#     key_q_list = ['crime', 'des']
#     key_c_list = ['crime', 'des']
# else:
#     pass
#
# # --- output path ---
# models_for_selection = ['lawformer',
#                        'bert-base-chinese',
#                        'LegalBERT/xs',
#                        'LegalBERT/ms',
#                         'RoBERTa',
#                        'longformer-base-4096',
#                        'longformer-large-4096'
#                        ]
#
# if not isSelectModel:
#     comparison = '3-Text'
#     selec = 0 # lawformer
#
# else:
#     comparison = '3-Text' # '2-Model'
#     selec = 1 # bert-base-chinese
#     # selec = 2 # xs-bert
#     # selec = 3 # ms-bert
#     # selec = 4 # RoBERTa
#     # selec = 5 # long-base
#     # selec = 6 # long-large
#     # print(f'models_for_selction： {models_for_selction}')
#     # selc = int(input('Input select number:'))
#
#
# selected_model = models_for_selection[selec]#'lawformer' # default
#
# if '/xs' in selected_model:
#     mode_name = 'xs_'
# elif'/ms' in selected_model:
#     mode_name = 'ms_'
# else:
#     mode_name = selected_model + '_'
# print(f'\nselected_model: {selected_model}\n')
#
# output_dir = f'./log/{mode_name}{signal}_results'
# model_save_path = f'./log/{mode_name}{signal}_model'
# validation_path = f'./log/{mode_name}{signal}_results/evaluation_results.json'
# output_file = os.path.join(output_path,f'{comparison}', f'{mode_name}{signal}.json')
#
# for path in [output_dir, model_save_path, validation_path, output_file]:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#
def getConcText(dict, key_list):
    conc_text = ""
    for attr in key_list:
        if attr in dict.keys():
            conc_text = conc_text + str(dict[attr])
    return conc_text


class LawMatchingDataset(Dataset):

    def __init__(self, ridx_list, tokenizer, max_length=512, mode='train'):
        self.ridx_list = ridx_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        for ridx in self.ridx_list:

            # --- get query text --
            qr = getQueryDict(ridx)
            query_text = getConcText(qr, key_q_list) # qr['q']

            # # === add SCR Feature ===
            # if isAddSCR or isAddFull or isFullText:
            #     try:
            #         query_text = str(qr['crime']) + str(qr['q']) + str(qr['des'])
            #     except:
            #         query_text = str(qr['crime']) + str(qr['q'])
            #
            # # --- only add crime ---
            # elif isAddCrime:
            #     try:
            #         query_text = str(qr['crime']) + str(qr['q'])
            #     except:
            #         query_text = str(qr['crime']) + str(qr['q'])
            # else:
            #     pass

            all_cids = get_all_cid(ridx)

            if self.mode == 'train':
                top30_data = top30_json.get(str(ridx), {})
                golden_cids = golden_json.get(str(ridx), [])

                for cid, similarity_level in top30_data.items():

                    # === get candidate text ===
                    cid = int(cid)
                    cd = getCandidateDict(ridx, cid)
                    candidate_text = getConcText(cd, key_c_list) #cd['ajjbqk']
                    # try:
                    #     if '以' not in cd['crime']:
                    #         if isAddSCR:
                    #             candidate_text = str(cd['crime']) + str(cd['ajjbqk'])
                    #         elif isAddCrime:
                    #             candidate_text = str(cd['ajjbqk']) + str(cd['crime'])
                    #         elif isAddFull:
                    #             candidate_text = str(cd['crime']) + str(cd['ajjbqk'])  + str(cd['pjjg'])
                    #         elif isFullText:
                    #             candidate_text = cd['qw']
                    #         else:
                    #             pass
                    # except Exception as e:
                    #     print(f'(cid={cid}, ridx={ridx}){e}')

                    # construct：query + candidate -> similarity degree
                    self.samples.append({
                        'ridx': ridx,
                        'cid': cid,
                        'query_text': query_text,
                        'candidate_text': candidate_text,
                        'label': similarity_level,  # 0-3
                        'is_golden': cid in golden_cids  # is relevant
                    })

                # negative sampling
                non_top30_cids = [cid for cid in all_cids if str(cid) not in top30_data]
                if non_top30_cids:
                    num_negative = min(len(top30_data), len(non_top30_cids))
                    negative_cids = random.sample(non_top30_cids, num_negative)

                    for cid in negative_cids:

                        # === get candidate text ===
                        cd = getCandidateDict(ridx, cid)
                        candidate_text = getConcText(cd, key_c_list) #cd['ajjbqk']
                        # try:
                        #     if '以' not in cd['crime']:
                        #         if isAddSCR:
                        #             candidate_text = str(cd['crime']) + str(cd['ajjbqk'])
                        #         elif isAddCrime:
                        #             candidate_text = str(cd['ajjbqk']) + str(cd['crime'])
                        #         elif isAddFull:
                        #             candidate_text = str(cd['crime']) + str(cd['ajName']) + \
                        #                              str(cd['ajjbqk'])  + \
                        #                              str(cd['writName']) + str(cd['pjjg'])
                        #         elif isFullText:
                        #             candidate_text = cd['qw']
                        #         else:
                        #             pass
                        # except Exception as e:
                        #     print(f'(cid={cid}, ridx={ridx}){e}')
                        #     # dict_keys(['ajId', 'ajName', 'ajjbqk', 'cpfxgc', 'pjjg', 'qw', 'writId', 'writName', 'cid'])
                        #     if isAddFull:
                        #         candidate_text = str(cd['ajName']) + \
                        #                          str(cd['ajjbqk'])  + \
                        #                          str(cd['writName']) + str(cd['pjjg'])
                        self.samples.append({
                            'ridx': ridx,
                            'cid': cid,
                            'query_text': query_text,
                            'candidate_text': candidate_text,
                            'label': 0,  # set negative as degree-0 of similarity
                            'is_golden': False
                        })

            else: # ference
                for cid in all_cids:
                    # === get candidate text ===
                    cd = getCandidateDict(ridx, cid)
                    candidate_text = getConcText(cd, key_c_list) # cd['ajjbqk']
                    # try:
                    #     try_get_crime = ""
                    #     # try_get_cpfxgc = ""
                    #
                    #     if 'crime' in cd.keys():
                    #         try_get_crime = cd['crime']
                    #     # if 'cpfxgc' in cd.keys():
                    #     #     try_get_cpfxgc = cd['cpfxgc']
                    #
                    #     if '以' not in try_get_crime:
                    #         if isAddSCR:
                    #             candidate_text = str(try_get_crime) + str(cd['ajjbqk'])
                    #         elif isAddCrime:
                    #             candidate_text = str(cd['ajjbqk']) + str(try_get_crime)
                    #         elif isAddFull:
                    #             candidate_text = str(try_get_crime) + str(cd['ajName']) + \
                    #                              str(cd['ajjbqk'])  + \
                    #                              str(cd['writName']) + str(cd['pjjg'])
                    #         elif isFullText:
                    #             candidate_text = cd['qw']
                    #         else:
                    #             pass
                    # except Exception as e:
                    #     print(f'(cid={cid}, ridx={ridx}){e}')
                    #     # dict_keys(['ajId', 'ajName', 'ajjbqk', 'cpfxgc', 'pjjg', 'qw', 'writId', 'writName', 'cid'])
                    #     if isAddFull:
                    #         candidate_text = str(cd['ajName']) + \
                    #                          str(cd['ajjbqk'])  \
                    #                         #str(cd['pjjg']) #str(cd['cpfxgc']) str(cd['writName']) +
                    self.samples.append({
                        'ridx': ridx,
                        'cid': cid,
                        'query_text': query_text,
                        'candidate_text': candidate_text,
                        'label': -1,
                        'is_golden': False
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # concatenate query-candidate pairs
        text = f"{sample['query_text']} [SEP] {sample['candidate_text']}"

        # encoding the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'ridx': sample['ridx'],
            'cid': sample['cid']
        }

        # add labels when training
        if self.mode == 'train':
            item['labels'] = torch.tensor(sample['label'], dtype=torch.long)

        return item


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'f1_macro': f1_macro
    }


def train_lawformer_matching():
    # print("Start to train_lawformer_matching...")

    # load the tokenizer
    model_name = selected_model # "lawformer"
    input_model_path = os.path.join(os.getcwd(), 'models', model_name)
    tokenizer = BertTokenizer.from_pretrained(input_model_path)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # split ridx
    all_ridx = [q_dict['ridx'] for q_dict in all_query_dict]
    random.shuffle(all_ridx)

    split_idx = int(0.8 * len(all_ridx))
    train_ridx = all_ridx[:split_idx]
    val_ridx = all_ridx[split_idx:]

    print(f"len(all_ridx): {len(all_ridx)}")
    print(f"len(train_ridx): {len(train_ridx)}")
    print(f"len(val_ridx): {len(val_ridx)}")

    # create dataset
    train_dataset = LawMatchingDataset(train_ridx, tokenizer, mode='train')
    val_dataset = LawMatchingDataset(val_ridx, tokenizer, mode='train')

    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(val_dataset): {len(val_dataset)}")

    # label distribution
    train_labels = [sample['label'] for sample in train_dataset.samples]
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for label in train_labels:
        label_counts[label] += 1

    print("Training label distribution:")
    for level, count in label_counts.items():
        print(f"  Level-{level}: {count} samples")

    # load the model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            input_model_path,
            num_labels=4,  # 4-tier
            ignore_mismatched_sizes=True
        )
        print("✓ Load the model successsfully.")
    except Exception as e:
        print(f"! Error | Failed to load model: {e}")
        from transformers import BertConfig, BertForSequenceClassification
        config = BertConfig.from_pretrained(input_model_path)
        config.num_labels = 4
        model = BertForSequenceClassification(config)

    model.to(device)

    # --- set training parameters ---
    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        seed=42,
        report_to=None,
        dataloader_pin_memory=False
    )

    # create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # strart training
    print("strart training...")
    train_result = trainer.train()

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"✓ model_save_path: {model_save_path}")
    print(f"train_result.training_loss: {train_result.training_loss:.4f}")

    # validation
    eval_results = trainer.evaluate()
    print("Validation:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    return model_save_path


def predict_and_save(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    #  inference_dataset（with all ridx）
    inference_dataset = LawMatchingDataset(all_ridx, tokenizer, mode='inference')
    print(f"len(inference_dataset): {len(inference_dataset)}")

    # predicting
    predictions_by_ridx = defaultdict(list)

    with torch.no_grad():
        for i, sample in enumerate(inference_dataset.samples):
            if i % 1000 == 0:
                print(f"Processing: {i}/{len(inference_dataset)}")

            # Prepare input
            text = f"{sample['query_text']} [SEP] {sample['candidate_text']}"
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            ).to(device)

            # Predicting
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Calculating Similarity（of Level-3）
            similarity_score = predictions[0][3].item()

            predictions_by_ridx[sample['ridx']].append({
                'cid': sample['cid'],
                'score': similarity_score
            })

    # ranking
    final_predictions = {}
    for ridx, candidates in predictions_by_ridx.items():
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        # save cid list
        final_predictions[ridx] = [candidate['cid'] for candidate in sorted_candidates]


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_predictions, f, indent=2, ensure_ascii=False)

    print(f"✓ output_file: {output_file}")

    # Info
    print("\nInfo:")
    for ridx in list(final_predictions.keys())[:5]:
        cids = final_predictions[ridx]
        golden_cids = golden_json.get(str(ridx), [])
        top30_cids = list(top30_json.get(str(ridx), {}).keys())
        top30_cids = [int(cid) for cid in top30_cids]

        top30_pred = cids[:30]
        golden_in_top30 = len(set(top30_pred) & set(golden_cids))

        print(
            f"ridx {ridx}: len(cids) = {len(cids)}, len(golden_cids) = {len(golden_cids)}, len(set(top30_pred) & set(golden_cids)) = {len(set(top30_pred) & set(golden_cids))}")


def test_lawformer(isTest=False):
    model_name = "lawformer"
    input_model_path = os.path.join(os.getcwd(), 'models', model_name)

    tokenizer = BertTokenizer.from_pretrained(input_model_path)

    if isTest:
        test_texts = [
            "滚滚长江东逝水",
            "被告人王某因盗窃罪被判处有期徒刑三年",
            "根据《中华人民共和国刑法》第二百六十四条规定",
        ]
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            print(f"'{text}'")
            print(f"  Tokens: {tokens}")
            print(f"  Token IDs: {tokenizer.encode(text, add_special_tokens=False)}")
            print("-" * 50)

        # Explore data
        print("\nxplore data:")
        print(f"len(all_ridx): {len(all_ridx)}")

        sample_ridx = random.choice(all_ridx)
        print(f"\nExample ridx: {sample_ridx}")
        print(f"Query: {getQueryDict(sample_ridx)['q']}")

        all_cids = get_all_cid(sample_ridx)
        print(f"len(all_cids): {len(all_cids)}")

        golden_cids = golden_json.get(str(sample_ridx), [])
        print(f"len(golden_cids): {len(golden_cids)}")

        top30_data = top30_json.get(str(sample_ridx), {})
        print(f"len(top30_data): {len(top30_data)}")

        # Distribution of degrees in top30
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for cid, level in top30_data.items():
            level_counts[level] += 1
        print("Top30 degree distribution:", level_counts)
        return

    # Start training
    print("Start training...")
    model_path = train_lawformer_matching()

    # Start prediction and save results
    predict_and_save(model_path)


# if __name__ == "__main__":
#     # testing
#     # test_lawformer(True)
#
#     # training
#     test_lawformer(False)
#     base_path = os.path.join(os.getcwd(), 'data')
#     visualConfig(isVisualAll=False, pid=3)

# === All models ===
comparison = '3-Text'
models_for_selection = [
                        # 'lawformer',
                        # 'bert-base-chinese',
                        'LegalBERT/xs',
                        'LegalBERT/ms',
                        'RoBERTa',
                        'longformer-base-4096',
                        'longformer-large-4096'
                        ]

modes_for_selection = [
                        'Regular',
                        'isAddSKR',
                        'isAddCrime',
                        'isAddFull',
                        'isFullText',
                        'isFourTier',
                        'isFourTierOnly'
                        ]

def getModeSetup(mode):

    if mode == 'isAddSKR':
        signal = 'SKR_'
        key_q_list = ['crime', 'q', 'des']
        key_c_list = ['crime', 'ajjbqk']

    elif mode == 'isAddCrime':
        signal = 'Simplified_'
        key_q_list = ['q', 'crime']
        key_c_list = ['ajjbqk', 'crime']

    elif mode == 'isAddFull':
        signal = 'Full_'
        key_q_list = ['crime', 'q', 'des']
        key_c_list = ['crime', 'ajjbqk','cpfxgc','pjjg']

    elif mode == 'isFullText':
        signal = 'SimpleFullText_'
        key_q_list = ['crime', 'q', 'des']
        key_c_list = ['qw']

    elif mode == 'isFourTier':
        signal = 'FourTier_'
        key_q_list = ['crime', 'q', 'des']
        key_c_list = ['crime', 'ajjbqk', 'des']

    elif mode == 'isFourTierOnly':
        signal = 'FourTierOnly_'
        key_q_list = ['crime', 'des']
        key_c_list = ['crime', 'des']

    else: # Regular
        signal = 'Regular_'
        key_q_list = ['q']
        key_c_list = ['ajjbqk']

    return signal, key_q_list, key_c_list

# === ALL MODELS ====
for model in models_for_selection:
    selected_model = model
    # --- path name ---
    if '/xs' in selected_model:
        mode_name = 'xs_'
    elif '/ms' in selected_model:
        mode_name = 'ms_'
    else:
        mode_name = selected_model + '_'
    print(f'\nselected_model: {selected_model}\n')

    # === ALL MODES ===
    for mode in modes_for_selection:
        print(f'\nmode:{mode}\n')
        signal, key_q_list, key_c_list =getModeSetup(mode)

        # --- paths ---
        output_dir = f'./log/{mode_name}{signal}_results'
        model_save_path = f'./log/{mode_name}{signal}_model'
        validation_path = f'./log/{mode_name}{signal}_results/evaluation_results.json'
        output_file = os.path.join(output_path, f'{comparison}', f'{mode_name}{signal}.json')

        for path in [output_dir, model_save_path, validation_path, output_file]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # --- main ---
        test_lawformer(False)

visualConfig(isVisualAll=False, pid=3)