### mistral model
# conda activate cuda
# pip install --upgrade git+https://github.com/UKPLab/sentence-transformers
# pip install keybert ctransformers[cuda]
# pip install --upgrade git+https://github.com/huggingface/transformers
# python jurex_mistral.py

# --- test setup ---
from processing import *
isTestLLM = False
isTestCrimeRec = True

# --- mistral model ---
from ctransformers import AutoModelForCausalLM

model_type = "mistral"
model_files = [
    "mistral-7b-instruct-v0.1.Q2_K.gguf",
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
]

fid = 0 # mistral-Q2
# fid = 1 # mistral-Q4


# --- init mistral model ---
model = AutoModelForCausalLM.from_pretrained(
    "E:/Data/models/",
    model_file= model_files[fid],
    model_type= model_type,
    gpu_layers=50,
    hf=False,
    lib='avx2',  #'cuda',
    context_length=2048,
    max_new_tokens=1024,
    temperature=0.5,
    top_p=0.85,
    repetition_penalty=1.2,
    stop=["<|endoftext|>", "<|im_end|>", "</s>", "###"],
    top_k=40,
    seed=42
)

# --- adaption CN ---
def qa(prompt):
    # restriction
    prompt_des = """当prompt为中文时，除了特定计算机、数学的专有名词之外，只使用中文回答。"""
    formatted_prompt = f"""<s>[INST] {prompt_des} {prompt} [/INST]"""

    # response
    response = model(
        formatted_prompt,
        stream=False,
        context_length=2048,
        max_new_tokens=1024
    )
    return response

# --- test LLM ---
if isTestLLM:
    test_en = 'Please explain the construction factors of the theft crime.'
    test_cn = '请解释盗窃罪的构成要件'
    print(qa(test_cn))

if isTestCrimeRec:
    # crime_prompt = crime_rec + crime_list + test_q # test_c
    crime_prompt = '【问题】 请分析以下案例构成何罪名：' + '【案例】' + test_q[:200]
    print(qa(crime_prompt))