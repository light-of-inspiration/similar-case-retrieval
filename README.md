# similar-case-retrieval
## Introduction  
Similar case retrieval (SCR) task is a specific text-classification task especially in legal, medical and other related domians.  

This project realised a lawformer-based method with Semantic Keyword Representaion (SKR) module that exceeding the baseline of BM25, TF-IDF and BERT.  

The experiment is based on the comparsion of performance with the LeCaRD dataset.  
  
  
## Project: Similar Case Retrieval Method based on the Pre-trained Model  
Author: W. LIN  

### I. Code Structure  
**(A) data:** the LeCaRD dataset folder  
    (i) query: query.json offer a dict with 107 queries.   
    (ii) candidates: each query with an ridx, has an according folder of candidates, named by cid.json (please re-download the full version: https://github.com/myx666/LeCaRD)  
    (iii) label: golden_labels.json (shows relevant cids); label_top30_dict_json (show top-30 most similar cases' id and the degree from 0-3)   
    (iv) prediction (test_prediction): the work (test) prediction output path, where the visualize.py get the data sources.  
    (v) others: contains stopword.txt, etc.  
    
**(B) models:** local models like (please re-download them here):  
    (i)   lawformer (https://huggingface.co/thunlp/Lawformer)  
    (ii)  bert-base-chinese  
    (iii) bert/xs (https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/xs.zip)  
    (iv)  bert/ms (https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/ms.zip)  
    (v)   RoBERTa_zh_L12_PyTorch   
    (vi)  longformer-base-4096  
    (vii) longformer-large-4096  
    ……  

**(C) python files:**  
The CAIL contest has offered dataset, baseline.py (BM25), metrics.py, stopword.txt/  
As the given original baseline would be inefficient, not structured, and not easy to understand with unstructured organisation.  

Thus, I reconstruct the BM25 baseline and metric program into more structured style,  
including file processing parts, similarity calculation part, and evaluation parts.  

For comparison, I add different similarity function in similarity.py, and the multi-thread mechanism for acceleration.  
To use lawformer, run the 'utilize.py' file with GPU acceleration.  

==========================================  
Here is introductions for main python files:  

**1.processing.py**
    The structured json processing functions, necessary almost for all other executor.  
    Use 'from processing import *' to get all available json funcions.  
    Here are 2 global variable 'isKaggle' and 'isTestProcess'.  
    If you are using Kaggle notebook, set 'isKaggle=True' to access online models.  
    If you want to access local models, set 'isKaggle=False' and right model paths.  
    The variable 'isTestProcess' is used to test the functions of the 'process.py'.  
    If you are using for the first time, set 'isTestProcess=True' to check all the file processing functions.  

**2.similarity.py**  
    This file defines different similarity calculation methods.  
    As the BM25 version offered by CAIL has some version problem, here gives the version of Okapi BM25.  
    I also define the functions to get TF-IDF, Jaccard, etc. similarities.  


**3.similarity_acc.py**  
    This file call the similarity functions and get predictions with multi-thread mechanism.  
    The upper __main__ function can realise TF-IDF, BM25 results.  
    The lower __main__ function can get BERT prediction.  

**4. jurex_XXX.py**  
    If you want to use the feature extraction, download models and reset the path:  
    (i)  qwen1.5-7b-chat-q4_k_m.gguf   
    (ii) Qwen-7B-Chat.Q4_K_M.gguf  
    (iii)mistral-7b-instruct-v0.1.Q2_K.gguf   
    (iv) mistral-7b-instruct-v0.1.Q4_K_M.gguf   
    (iv) DeepSeek-R1-Distill-Qwen-7B-Uncensored.i1-Q4_K_S.gguf   
    The jurex_XXX.py code show how to test the extraction ability with this model.   
    We use ./JUREX/data/flattened_jurex4e.json to assist the extraction.   
    The extraction process is also in the processing.py   

**5.ultilize.py**   
    This file realise the lawformer method. Detailed methodology is given in the thesis.   
    * If the environment is wrong, try to run 'zero_env.py' to repair.   

**6. visualize.py**  
    Select a folder (test_)/prediction and to get all the json predictions for evaluation.  
    * Please check the path of 'data' folder and the sub 'prediction/2-Model' folder  
    * The given results in the folder can be visualized like this:  

```
                Model NDCG@10 NDCG@20 NDCG@30    P@5   P@10    MAP  
              BERT_ms  0.7210  0.7784  0.8764 0.3800 0.3790 0.4369  
              BERT_xs  0.7103  0.7764  0.8706 0.3700 0.3720 0.4271  
              RoBERTa  0.7255  0.7825  0.8769 0.3820 0.3860 0.4426  
    bert-base-chinese  0.7820  0.8372  0.9068 0.4180 0.4160 0.4851  
                 bm25  0.6766  0.7591  0.8659 0.3120 0.3010 0.3790  
            lawformer  0.8807  0.9074  0.9499 0.6120 0.5370 0.7393  
 longformer-base-4096  0.7169  0.7807  0.8763 0.3740 0.3670 0.4220  
longformer-large-4096  0.7022  0.7693  0.8669 0.3660 0.3570 0.4201  
```
  
**7. SHAP.py**
    This file illustrate the feature importance.  
    * If the environment is wrong, try to run 'SHAP.py' to repair.  

  
==========================================  
**II. Environment settings of the author:**
(The requirements.txt etc are provided)  

```bash
(acc) >python --version
Python 3.10.19

(acc) >conda list
```
  
packages in environment at E:\Data\Anaconda3\envs\acc:  
```
Name                    Version                   Build  Channel
accelerate                1.11.0                   pypi_0    pypi  
aiohappyeyeballs          2.6.1                    pypi_0    pypi  
aiohttp                   3.13.2                   pypi_0    pypi  
aiosignal                 1.4.0                    pypi_0    pypi  
anyio                     4.11.0                   pypi_0    pypi  
async-timeout             5.0.1                    pypi_0    pypi  
attrs                     25.4.0                   pypi_0    pypi  
blas                      1.0                         mkl    defaults  
bzip2                     1.0.8                h2bbff1b_6    defaults  
ca-certificates           2025.11.4            haa95532_0    defaults  
certifi                   2025.11.12               pypi_0    pypi  
charset-normalizer        3.4.4                    pypi_0    pypi  
colorama                  0.4.6                    pypi_0    pypi  
datasets                  4.4.1                    pypi_0    pypi  
dill                      0.4.0                    pypi_0    pypi  
exceptiongroup            1.3.0                    pypi_0    pypi  
expat                     2.7.3                h9214b88_0    defaults  
filelock                  3.20.0                   pypi_0    pypi  
frozenlist                1.8.0                    pypi_0    pypi  
fsspec                    2025.10.0                pypi_0    pypi  
gensim                    4.4.0                    pypi_0    pypi  
h11                       0.16.0                   pypi_0    pypi  
httpcore                  1.0.9                    pypi_0    pypi  
httpx                     0.28.1                   pypi_0    pypi  
huggingface-hub           0.36.0                   pypi_0    pypi  
icc_rt                    2022.1.0             h6049295_2    defaults  
idna                      3.11                     pypi_0    pypi  
intel-openmp              2025.0.0          haa95532_1164    defaults  
jieba                     0.42.1                   pypi_0    pypi  
jinja2                    3.1.6                    pypi_0    pypi  
joblib                    1.5.2           py310haa95532_0    defaults  
libffi                    3.4.4                hd77b12b_1    defaults  
libzlib                   1.3.1                h02ab6af_0    defaults  
markupsafe                3.0.3                    pypi_0    pypi  
mkl                       2025.0.0           h5da7b33_930    defaults  
mkl-service               2.5.2           py310h0b37514_0    defaults  
mkl_fft                   2.1.1           py310h300f80d_0    defaults  
mkl_random                1.3.0           py310ha5e6156_0    defaults  
mpmath                    1.3.0                    pypi_0    pypi  
multidict                 6.7.0                    pypi_0    pypi  
multiprocess              0.70.18                  pypi_0    pypi  
networkx                  3.4.2                    pypi_0    pypi  
numpy                     2.2.6                    pypi_0    pypi  
numpy-base                2.2.5           py310h7794460_2    defaults  
nvidia-cublas             13.1.0.3                 pypi_0    pypi  
nvidia-cudnn-cu13         9.13.0.50                pypi_0    pypi  
openssl                   3.0.18               h543e019_0    defaults  
packaging                 25.0                     pypi_0    pypi  
pandas                    2.3.3                    pypi_0    pypi  
pillow                    12.0.0                   pypi_0    pypi  
pip                       25.3               pyhc872135_0    defaults  
propcache                 0.4.1                    pypi_0    pypi  
protobuf                  3.20.3                   pypi_0    pypi  
psutil                    7.1.3                    pypi_0    pypi  
pyarrow                   22.0.0                   pypi_0    pypi  
python                    3.10.19              h981015d_0    defaults  
python-dateutil           2.9.0.post0              pypi_0    pypi  
python_abi                3.10                    2_cp310    https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge  
pytz                      2025.2                   pypi_0    pypi  
pyyaml                    6.0.3                    pypi_0    pypi  
rank-bm25                 0.2.2                    pypi_0    pypi  
regex                     2025.11.3                pypi_0    pypi  
requests                  2.32.5                   pypi_0    pypi  
safetensors               0.6.2                    pypi_0    pypi  
scikit-learn              1.7.2           py310h21054b0_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge  
scipy                     1.15.3          py310h1bbe36f_1    defaults  
setuptools                80.9.0          py310haa95532_0    defaults  
six                       1.17.0                   pypi_0    pypi  
smart-open                7.5.0                    pypi_0    pypi  
sniffio                   1.3.1                    pypi_0    pypi  
sqlite                    3.51.0               hda9a48d_0    defaults  
sympy                     1.14.0                   pypi_0    pypi  
tbb                       2022.0.0             h214f63a_0    defaults  
tbb-devel                 2022.0.0             h214f63a_0    defaults  
threadpoolctl             3.5.0           py310h9909e9c_0    defaults  
tk                        8.6.15               hf199647_0    defaults  
tokenizers                0.22.1                   pypi_0    pypi  
torch                     2.9.0+cu130              pypi_0    pypi  
torchaudio                2.9.0+cu130              pypi_0    pypi  
torchvision               0.24.0+cu130             pypi_0    pypi   
tqdm                      4.67.1                   pypi_0    pypi  
transformers              4.57.1                   pypi_0    pypi  
typing-extensions         4.15.0                   pypi_0    pypi  
tzdata                    2025.2                   pypi_0    pypi  
ucrt                      10.0.22621.0         haa95532_0    defaults  
urllib3                   2.5.0                    pypi_0    pypi  
vc                        14.3                h2df5915_10    defaults  
vc14_runtime              14.44.35208         h4927774_10    defaults  
vs2015_runtime            14.44.35208         ha6b5a95_10    defaults  
wheel                     0.45.1          py310haa95532_0    defaults  
wrapt                     2.0.1                    pypi_0    pypi   
xxhash                    3.6.0                    pypi_0    pypi   
xz                        5.6.4                h4754444_1    defaults  
yarl                      1.22.0                   pypi_0    pypi   
zlib                      1.3.1                h02ab6af_0    defaults  
```
