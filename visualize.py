import os
import json
import math
import numpy as np
from typing import Dict, List, Any, Tuple
import pandas as pd

from processing import *

class LegalRetrievalEvaluator:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.label_path_top30 = os.path.join(base_path, 'label', 'label_top30_dict.json')

    def load_predictions(self, pred_folder: str = 'prediction') -> Dict[str, Any]:
        # Labels
        with open(self.label_path_top30, 'r') as file:
            avglist = json.load(file)

        # Predictions
        pred_path = os.path.join(self.base_path, pred_folder)
        pred_files = [f for f in os.listdir(pred_path) if f.endswith('.json')]
        predictions = {}

        for pred_file in pred_files:
            file_path = os.path.join(pred_path, pred_file)
            model_name = pred_file.replace('.json', '')

            try:
                if model_name == 'bert':
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    predictions[model_name] = [eval(line) for line in lines]
                else:
                    with open(file_path, 'r') as f:
                        predictions[model_name] = json.load(f)

                if model_name in ['tfidf_top100', 'bm25_top100']:
                    for key in list(predictions[model_name].keys())[:100]:
                        predictions[model_name][key].reverse()

            except Exception as e:
                print(f"Errors | failed to load (model_name: {model_name}): {e}")
                continue

        return {
            'labels': avglist,
            'predictions': predictions
        }

    def kappa(self, testData: List[List[int]], k: int) -> float:
        """Cohen's Kappa coefficient"""
        dataMat = np.mat(testData)
        P0 = 0.0
        for i in range(k):
            P0 += dataMat[i, i] * 1.0
        xsum = np.sum(dataMat, axis=1)
        ysum = np.sum(dataMat, axis=0)
        Pe = float(ysum * xsum) / k ** 2
        P0 = float(P0 / k * 1.0)
        cohens_coefficient = float((P0 - Pe) / (1 - Pe))
        return cohens_coefficient

    def fleiss_kappa(self, testData: List[List[int]], N: int, k: int, n: int) -> float:
        """Fleiss Kappa coefficient"""
        dataMat = np.mat(testData, float)
        oneMat = np.ones((k, 1))
        total_sum = 0.0
        P0 = 0.0

        for i in range(N):
            temp = 0.0
            for j in range(k):
                total_sum += dataMat[i, j]
                temp += 1.0 * dataMat[i, j] ** 2
            temp -= n
            temp /= (n - 1) * n
            P0 += temp

        P0 = 1.0 * P0 / N
        ysum = np.sum(dataMat, axis=0)

        for i in range(k):
            ysum[0, i] = (ysum[0, i] / total_sum) ** 2

        Pe = ysum * oneMat * 1.0
        ans = (P0 - Pe) / (1 - Pe)
        return ans[0, 0]

    def ndcg(self, ranks: List[float], K: int) -> float:
        """NDCG@K"""
        dcg_value = 0.0
        idcg_value = 0.0
        sranks = sorted(ranks, reverse=True)

        for i in range(K):
            logi = math.log(i + 2, 2)
            dcg_value += ranks[i] / logi
            idcg_value += sranks[i] / logi

        return dcg_value / idcg_value if idcg_value > 0 else 0.0

    def evaluate_ndcg(self, predictions: Dict, labels: Dict, query_keys: List[str], topK_list: List[int]) -> Dict[
        str, List[float]]:
        """Evaluating NDCG indicators"""
        results = {}

        for model_name, pred_dict in predictions.items():
            if model_name == 'bert':
                for i, bert_pred in enumerate(pred_dict):
                    model_key = f'bert_{i + 1}'
                    results[model_key] = []
                    for topK in topK_list:
                        sndcg = 0.0
                        for key in query_keys:
                            rawranks = [labels[key].get(str(i), 0) for i in bert_pred.get(key, [])
                                        if str(i) in labels[key]]
                            ranks = rawranks + [0] * (30 - len(rawranks))
                            if sum(ranks) != 0:
                                sndcg += self.ndcg(ranks, topK)
                        results[model_key].append(sndcg / len(query_keys))
            else:
                results[model_name] = []
                for topK in topK_list:
                    sndcg = 0.0
                    for key in query_keys:
                        rawranks = [labels[key].get(str(i), 0) for i in pred_dict.get(key, [])
                                    if str(i) in labels[key]]
                        ranks = rawranks + [0] * (30 - len(rawranks))
                        if sum(ranks) != 0:
                            sndcg += self.ndcg(ranks, topK)
                    results[model_name].append(sndcg / len(query_keys))

        return results

    def evaluate_precision(self, predictions: Dict, labels: Dict, query_keys: List[str], topK_list: List[int]) -> Dict[
        str, List[float]]:
        results = {}

        for model_name, pred_dict in predictions.items():
            if model_name == 'bert':
                for i, bert_pred in enumerate(pred_dict):
                    model_key = f'bert_{i + 1}'
                    results[model_key] = []
                    for topK in topK_list:
                        sp = 0.0
                        for key in query_keys:
                            ranks = [i for i in bert_pred.get(key, []) if str(i) in labels[key]]
                            sp += len([j for j in ranks[:topK] if labels[key].get(str(j), 0) == 3]) / topK
                        results[model_key].append(sp / len(query_keys))
            else:
                results[model_name] = []
                for topK in topK_list:
                    sp = 0.0
                    for key in query_keys:
                        ranks = [i for i in pred_dict.get(key, []) if str(i) in labels[key]]
                        sp += len([j for j in ranks[:topK] if labels[key].get(str(j), 0) == 3]) / topK
                    results[model_name].append(sp / len(query_keys))

        return results

    def evaluate_map(self, predictions: Dict, labels: Dict, query_keys: List[str]) -> Dict[str, float]:
        """MAP"""
        results = {}

        for model_name, pred_dict in predictions.items():
            if model_name == 'bert':
                for i, bert_pred in enumerate(pred_dict):
                    model_key = f'bert_{i + 1}'
                    smap = 0.0
                    for key in query_keys:
                        ranks = [i for i in bert_pred.get(key, []) if str(i) in labels[key]]
                        rels = [ranks.index(i) for i in ranks if labels[key].get(str(i), 0) == 3]
                        tem_map = 0.0
                        for rel_rank in rels:
                            tem_map += len([j for j in ranks[:rel_rank + 1] if labels[key].get(str(j), 0) == 3]) / (
                                        rel_rank + 1)
                        if len(rels) > 0:
                            smap += tem_map / len(rels)
                    results[model_key] = smap / len(query_keys)
            else:
                smap = 0.0
                for key in query_keys:
                    ranks = [i for i in pred_dict.get(key, []) if str(i) in labels[key]]
                    rels = [ranks.index(i) for i in ranks if labels[key].get(str(i), 0) == 3]
                    tem_map = 0.0
                    for rel_rank in rels:
                        tem_map += len([j for j in ranks[:rel_rank + 1] if labels[key].get(str(j), 0) == 3]) / (
                                    rel_rank + 1)
                    if len(rels) > 0:
                        smap += tem_map / len(rels)
                results[model_name] = smap / len(query_keys)

        return results

    def evaluate_kappa(self) -> float:
        """Kappa consistency"""
        label_path = self.label_path_top30
        lists = json.load(open(label_path, 'r'))
        dataArr = []

        for i in lists.keys():
            for j in range(30):
                tem = [0, 0, 0, 0]
                for k in range(3):
                    # tem[int(lists[k][i][j])-1] += 1
                    pass
                dataArr.append(tem)

        return self.fleiss_kappa(dataArr, 3000, 4, 3)

    def get_query_keys(self, query_set: str = 'all', predictions: Dict = None) -> List[str]:
        """Get query keys"""
        if not predictions:
            return []

        ref_model = next((model for model in predictions.keys() if model != 'bert'), None)
        if not ref_model:
            return []

        all_keys = list(predictions[ref_model].keys())[:100]

        if query_set == 'all':
            return all_keys
        elif query_set == 'common':
            return all_keys[:77]
        elif query_set == 'controversial':
            return all_keys[77:100]
        elif query_set == 'test':
            return [key for i, key in enumerate(all_keys) if i % 5 == 0]
        else:
            return all_keys

    def evaluate_all_metrics(self, pred_folder: str = 'prediction', query_set: str = 'all') -> Dict[str, Any]:
        # loading data
        data = self.load_predictions(pred_folder)
        labels = data['labels']
        predictions = data['predictions']

        # get query keys
        query_keys = self.get_query_keys(query_set, predictions)

        # results
        results = {}

        # NDCG
        ndcg_results = self.evaluate_ndcg(predictions, labels, query_keys, [10, 20, 30])
        results['NDCG'] = ndcg_results

        # Precision
        precision_results = self.evaluate_precision(predictions, labels, query_keys, [5, 10])
        results['Precision'] = precision_results

        # MAP
        map_results = self.evaluate_map(predictions, labels, query_keys)
        results['MAP'] = map_results

        return results

    def print_comparison_table(self, results: Dict[str, Any]):
        table_data = []

        # Model names
        all_models = set()
        for metric_name, metric_results in results.items():
            all_models.update(metric_results.keys())

        all_models = sorted(all_models)

        # Line data
        for model in all_models:
            row = {'Model': model}

            # NDCG
            if model in results['NDCG']:
                ndcg_scores = results['NDCG'][model]
                row['NDCG@10'] = f"{ndcg_scores[0]:.4f}"
                row['NDCG@20'] = f"{ndcg_scores[1]:.4f}"
                row['NDCG@30'] = f"{ndcg_scores[2]:.4f}"
            else:
                row['NDCG@10'] = 'N/A'
                row['NDCG@20'] = 'N/A'
                row['NDCG@30'] = 'N/A'

            # Precision
            if model in results['Precision']:
                precision_scores = results['Precision'][model]
                row['P@5'] = f"{precision_scores[0]:.4f}"
                row['P@10'] = f"{precision_scores[1]:.4f}"
            else:
                row['P@5'] = 'N/A'
                row['P@10'] = 'N/A'

            # MAP
            if model in results['MAP']:
                row['MAP'] = f"{results['MAP'][model]:.4f}"
            else:
                row['MAP'] = 'N/A'

            table_data.append(row)

        # DataFrame
        df = pd.DataFrame(table_data)
        # print("\n 模型性能比较表:")
        print("\nComparison Results:")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)


def main_evaluation(base_path, pred_folder, showKappa = False):
    # print(f'\npred_folder = {pred_folder}')
    evaluator = LegalRetrievalEvaluator(base_path)

    # results = evaluator.evaluate_all_metrics(pred_folder='prediction', query_set='all')
    results = evaluator.evaluate_all_metrics(pred_folder=pred_folder, query_set='all')

    evaluator.print_comparison_table(results)

    if showKappa:
        kappa_score = evaluator.evaluate_kappa()
        print(f"\nFleiss Kappa: {kappa_score:.4f}")



# --- mode config ---
isVisualAll = False
# isVisualAll = True

# --- config ---
pid = 1 # 1-Sim
# pid = 2 # 2-Model
# pid = 3 # 3-Text

def visualConfig(isVisualAll = False, pid = 1):
    pred_mode = [
        '1-Sim',
        '2-Model',
        '3-Text'
    ]

    # --- visualizing ---
    if isVisualAll:
        for mode in pred_mode:
            print(f'\nMODE： {mode}')
            pred_folder = os.path.join('prediction', mode)
            main_evaluation(data_path, pred_folder)
    else:
        # --- exec ---
        pid -= 1
        pred_folder = os.path.join('prediction', pred_mode[pid])
        print(f'\nMODE： {pred_mode[pid]}')
        main_evaluation(data_path, pred_folder)

visualConfig(isVisualAll, pid)
