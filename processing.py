# processing.py

import os
import json
import jieba

# from jurex_qwen import *

# --- Path ---
isKaggle, isTestProcess,  isTestJurex, showPlot = False, False, False, False

# isKaggle = True        # Online Models
# isTestProcess = True   # Test if processing.py works
# isTestJurex = True     # Test if jurex module works
showPlot = True       # Show dataset statistic plots

print(f'isKaggle: {isKaggle}')
print(f'isTestProcess: {isTestProcess}')

if isKaggle:
    data_path = '/kaggle/input/lajs-2021/' # the data folder
    stopword_path = '/kaggle/input/lajs-2021-stopword/stopword.txt'
    output_path = 'kaggle/working'
else:
    data_path = os.path.join(os.getcwd(), 'data')
    stopword_path = os.path.join(data_path, 'others', 'stopword.txt')
    output_path =  os.path.join(data_path,'prediction')

# --- Get query & candiadates ---
query_path = os.path.join(data_path,'query','query.json')
candidate_path = os.path.join(data_path, 'candidates')
model_path = os.path.join(data_path, 'models')

all_query = open(query_path, 'r', encoding='utf-8').readlines() # list, len:107
all_query_dict = [json.loads(line.strip()) for line in all_query]
all_ridx = [q_dict['ridx'] for q_dict in all_query_dict]

# get the query-dict with (ridx)
def getQueryDict(ridx):
    return next((d for d in all_query_dict if d.get('ridx') == ridx), None)

# get all the candiates with (ridx)
def getCandidatesJSONs(ridx):
    candidates_names = os.listdir(os.path.join(data_path,'candidates', str(ridx))) # all json file names
    candidates_json = []
    for candidate_name in candidates_names:
        candidate_path = os.path.join(data_path,'candidates', str(ridx), candidate_name)
        candidate_json = json.load(open(candidate_path, 'r', encoding='utf-8'))
        cid = int(candidate_name.split('.')[0])
        candidate_json['cid'] = cid
        candidates_json.append(candidate_json)
    return candidates_json

def getCandidateDict(ridx, cid):
    candidate_path = os.path.join(data_path,'candidates', str(ridx), str(cid) + '.json')
    candidate_json = json.load(open(candidate_path, 'r', encoding='utf-8'))
    candidate_json['cid'] = cid
    return candidate_json

def get_all_cid(ridx):
    names = os.listdir(os.path.join(candidate_path, str(ridx)))
    return [int(name.split('.')[0]) for name in names]


# --- Labels ---
label_path = data_path + '/label/'
label_path_golden = label_path + 'golden_labels.json'
label_path_top30 = label_path + 'label_top30_dict.json'

with open(label_path_golden, 'r', encoding='utf-8') as file:
    golden_json = json.load(file)

with open(label_path_top30, 'r', encoding='utf-8') as file:
    top30_json = json.load(file)

# --- Stopwords ---
def getStopwords(stopword_path):
    stopwords = []
    with open(stopword_path, 'r', encoding='utf-8') as g:
        words = g.readlines()
        stopwords = [i.strip() for i in words]
        stopwords.extend(['.', '（', '）', '-'])
    return stopwords

stopwords = getStopwords(stopword_path)

# --- Corpus ---
def getCorpus(text):
    cut = jieba.cut(text, cut_all=False)
    joined = " ".join(cut).split()
    corpus = [i for i in joined if not i in stopwords]
    return corpus


# test all processing functions
def test_processing(ridx=5156,cid=20010):
    print(f'''Testing processing functions, ridx = { ridx }  cid = { cid }
getQueryDict({ridx})           keys() = { getQueryDict(ridx).keys() }                                                        <class 'dict'>  
getCandidatesJSONs({ridx}) [0].keys() = { getCandidatesJSONs(ridx)[0].keys() }     <class 'list'>
getCandidateDict({ridx}, {cid})['ajName']= { getCandidateDict(ridx, cid)['ajName']  }
len(getCorpus(getCandidateDict{ridx}, {cid})['ajjbqk'])) = {len(getCorpus(getCandidateDict(ridx,cid)['ajjbqk']))}
len(get_all_cid({ridx})) = {len(get_all_cid(ridx))}
golden_json[str({ridx})] = {golden_json[str(ridx)]} 
top30_json[str({ridx})] = {top30_json[str(ridx)]}
len(stopwords) = {len(stopwords)}
''')

if isTestProcess:
    test_processing()

""" test_processing()        ridx = 5156  cid = 20010

getQueryDict(5156)           keys() = dict_keys(['path', 'ridx', 'q', 'crime'])                                                        <class 'dict'>
getCandidatesJSONs(5156) [0].keys() = dict_keys(['ajId', 'ajName', 'ajjbqk', 'cpfxgc', 'pjjg', 'qw', 'writId', 'writName', 'cid'])     <class 'list'>
getCandidateDict(5156, 20010)['ajName']= 杨金雄、杨某某寻衅滋事一案
len(getCorpus(getCandidateDict5156, 20010)['ajjbqk'])) = 402
len(get_all_cid(5156)) = 100
golden_json[str(5156)] = [32518, 36655, 501, 32377, 4348, 27033, 28530, 21312, 31607, 7859, 34099, 1970, 11940, 42565, 18097, 39991, 14776, 39608, 20875]
top30_json[str(5156)] = {'38633': 2, '38632': 2, '32518': 3, '36655': 3, '501': 3, '32377': 3, '4348': 3, '17848': 2, '27033': 3, '24364': 2, '28530': 3, '21312': 3, '31607': 3, '7859': 3, '34099': 3, '11977': 2, '1970': 3, '11940'
: 3, '42565': 3, '12976': 2, '28331': 2, '33175': 2, '18097': 3, '39991': 3, '38445': 2, '24091': 2, '14776': 3, '39608': 3, '20875': 3, '28626': 2}
len(stopwords) = 750

len(kg_crime_list)： 155

"""

# --- statistics ---

if showPlot:
    pass



# --- jurex crime names ---
def getCrimeList(ref_path):
    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        kg_crime_list = list(kg_data.keys())
        crime_list_str = "，".join(kg_crime_list)
        print(f"len(kg_crime_list)： {len(kg_crime_list)}")
        return crime_list_str
    except Exception as e:
        print(f"Error | Failed to load crime list： {e}")
        exit(1)

def JurexDes(j_path):
    with open(j_path, 'r', encoding='utf-8') as f:
        return json.load(f)  # <class 'dict'> len=155

jurex_path = os.path.join("E:/Py_Dev/IceBerg/JUREX/data", "flattened_jurex4e.json")
with open(jurex_path, 'r', encoding='utf-8') as f:
    jurex_des = json.load(f)  # <class 'dict'> len=155
crime_list = getCrimeList(jurex_path)
crime_rec = '''你是一个刑法领域的律师。请根据罪名列表，对于以下案情事实，请判断它包含了罪名列表中的哪些罪名。请注意，你只需要输出罪名，不要输出多余的信息，罪名必须在罪名列表中。罪名之间用‘，'隔开。",'''

test_ridx = 5156
test_cid = 20010
test_q = getQueryDict(test_ridx)['q']
test_c = getCandidateDict(test_ridx, test_cid)['ajjbqk']

if isTestJurex:
    print(f'crime_list： {crime_list}')
    # print(f'len(crime_list): {len(crime_list)}')



# --- jurex get 4-tier des ---
def get4tierDict(crime, text):
    # crime-dict-dict
    tier_dict = dict()
    if crime in jurex_des.keys():
        crime_des = jurex_des[crime] # dict_keys(['犯罪客体', '客观方面', '犯罪主体', '主观方面'])


    else:
        print('Non-exist crime.')


# --- add des ---


# --- check crime attr (candidate)
def checkCrimeAttr():
    finished_ridx = []
    for ridx in all_ridx:
        finished = 0
        cid_list = get_all_cid(ridx)
        ridx_dir = os.path.join(candidate_path, str(ridx))
        for cid in cid_list:
            c_dict = getCandidateDict(ridx, cid)
            try:
                crimes = c_dict['crime']
                if crimes: # 不为空
                    # print(f'cid={cid} （ridx={ridx}): has Crime {crimes}.')
                    finished += 1
                    # clean extract data
                    cleaned_data = cleanCandidateCrimes(crimes)
                    c_dict['crime'] = cleaned_data
                    # save json
                    cid_json_path = os.path.join(ridx_dir, f"{cid}.json")
                    with open(cid_json_path, 'w', encoding='utf-8') as f:
                        json.dump(c_dict, f, ensure_ascii=False, indent=2)
                    print(f"✅ cid={cid} (ridx={ridx})  {crimes} 已保存至: {cid_json_path}")
                    # exit()
            except:
                pass
        if finished >= len(cid_list):
            # print(f'ridx={ridx} finished.')
            finished_ridx.append(ridx)

    # print(f'finished: {len(finished_ridx)} \n{finished_ridx}')
    return finished_ridx

# f = checkCrimeAttr()
# print(f'\nProcessd q-c: {len(f)} \n{f}')

# --- clean candidate crimes extract ---

def cleanCandidateCrimes(text):
    """
    清洗大模型返回的crime字段，将其转换为标准化的list格式

    处理逻辑：
    1. 如果已经是list类型且包含的是字符串罪名，直接返回
    2. 如果list中包含的是字符串描述，提取其中的罪名
    3. 如果是字符串且包含标准的list格式，尝试提取并转换为list
    4. 其他情况返回空list

    参数:
        text: 输入的crime字段内容

    返回:
        list: 清洗后的罪名列表
    """
    # 如果输入是None或空值，返回空list
    if text is None:
        return []

    # 1. 如果已经是list类型
    if isinstance(text, list):
        # 如果list为空，直接返回
        if not text:
            return []

        # 检查list中的第一个元素是否是字符串
        if len(text) > 0 and isinstance(text[0], str):
            # 如果list中只有一个字符串元素，且该字符串包含罪名描述
            if len(text) == 1:
                content = text[0]
                # 如果这个字符串看起来像Python代码块或包含罪名列表
                if '```python' in content or '[' in content:
                    # 递归处理这个字符串
                    return cleanCandidateCrimes(content)
                # 如果直接是罪名，如"妨害公务罪"
                elif content.endswith('罪'):
                    return [content]
                else:
                    # 尝试从字符串中提取罪名
                    return extract_crimes_from_string(content)
            else:
                # 如果list中有多个元素，假设它们都是罪名
                cleaned_crimes = []
                for item in text:
                    if isinstance(item, str) and item.strip():
                        cleaned_crimes.append(item.strip())
                return cleaned_crimes
        else:
            # 如果不是字符串类型的list，返回空list
            return []

    # 2. 如果是字符串类型
    elif isinstance(text, str):
        cleaned_text = text.strip()

        # 如果为空字符串，返回空list
        if not cleaned_text:
            return []

        # 情况A: 直接是罪名，如"妨害公务罪"
        if cleaned_text.endswith('罪') and '[' not in cleaned_text:
            return [cleaned_text]

        # 情况B: 字符串已经是标准的Python list格式
        # 例如: "['抢劫罪', '聚众冲击国家机关罪', '容留他人吸毒罪']"
        try:
            # 尝试解析为list
            result = ast.literal_eval(cleaned_text)
            if isinstance(result, list):
                cleaned_items = []
                for item in result:
                    if isinstance(item, str) and item.strip():
                        cleaned_items.append(item.strip())
                if cleaned_items:
                    return cleaned_items
        except (ValueError, SyntaxError):
            pass

        # 情况C: 包含代码块标记
        # 例如: "```python\n[传播淫秽物品牟利罪', '制作、复制、出版、贩卖、传播淫秽物品牟利罪]\n```"
        if '```python' in cleaned_text or '```' in cleaned_text:
            # 提取代码块中的内容
            code_match = re.search(r'```python\s*(.*?)\s*```', cleaned_text, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\s*(.*?)\s*```', cleaned_text, re.DOTALL)

            if code_match:
                code_content = code_match.group(1).strip()
                # 尝试从代码块中提取list
                list_match = re.search(r'\[(.*?)\]', code_content, re.DOTALL)
                if list_match:
                    list_content = list_match.group(0)
                    try:
                        result = ast.literal_eval(list_content)
                        if isinstance(result, list):
                            cleaned_items = []
                            for item in result:
                                if isinstance(item, str) and item.strip():
                                    cleaned_items.append(item.strip())
                            if cleaned_items:
                                return cleaned_items
                    except (ValueError, SyntaxError):
                        # 如果无法直接eval，尝试手动提取
                        return extract_crimes_from_bracket_content(list_content)

        # 情况D: 包含方括号但不一定有代码块标记
        # 例如: "[传播淫秽物品牟利罪', '制作、复制、出版、贩卖、传播淫秽物品牟利罪]"
        if '[' in cleaned_text and ']' in cleaned_text:
            bracket_match = re.search(r'\[(.*?)\]', cleaned_text, re.DOTALL)
            if bracket_match:
                bracket_content = bracket_match.group(0)  # 获取整个方括号内容
                try:
                    # 尝试直接eval
                    result = ast.literal_eval(bracket_content)
                    if isinstance(result, list):
                        cleaned_items = []
                        for item in result:
                            if isinstance(item, str) and item.strip():
                                cleaned_items.append(item.strip())
                        if cleaned_items:
                            return cleaned_items
                except (ValueError, SyntaxError):
                    # 如果无法直接eval，尝试手动提取
                    return extract_crimes_from_bracket_content(bracket_content)

        # 情况E: 其他字符串，尝试提取所有罪名
        return extract_crimes_from_string(cleaned_text)

    # 3. 其他类型返回空list
    return []

def extract_crimes_from_bracket_content(bracket_content):
    """
    从方括号内容中提取罪名
    例如: "[传播淫秽物品牟利罪', '制作、复制、出版、贩卖、传播淫秽物品牟利罪]"
    """
    # 移除开头的'['和结尾的']'
    content = bracket_content.strip()[1:-1].strip()

    crimes = []

    # 方法1: 使用正则表达式匹配被引号包围的内容
    quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", content)
    if quoted_items:
        for item in quoted_items:
            if item.strip() and '罪' in item:
                # 提取罪名的核心部分（去除括号内容）
                crime_name = re.sub(r'[（\(].*?[）\)]', '', item).strip()
                if crime_name:
                    crimes.append(crime_name)

    # 方法2: 如果没有找到引号包围的内容，尝试按逗号分割
    if not crimes:
        items = content.split(',')
        for item in items:
            item = item.strip()
            if item and '罪' in item:
                # 清理可能的引号
                item = re.sub(r"^['\"]|['\"]$", '', item)
                # 提取罪名的核心部分
                crime_name = re.sub(r'[（\(].*?[）\)]', '', item).strip()
                if crime_name:
                    crimes.append(crime_name)

    return crimes

def extract_crimes_from_string(text):
    """
    从普通字符串中提取罪名
    """
    # 使用正则表达式匹配所有"XXX罪"格式的字符串
    crimes = re.findall(r'([\u4e00-\u9fa5]+罪)', text)

    # 去重并清理
    unique_crimes = []
    for crime in crimes:
        # 去除括号内容
        clean_crime = re.sub(r'[（\(].*?[）\)]', '', crime).strip()
        if clean_crime and clean_crime not in unique_crimes:
            unique_crimes.append(clean_crime)

    return unique_crimes


# --- execute clean candidate crimes extract ---
import ast
import re

# def cleanCandidateCrimes(text):
#     """
#     清洗大模型返回的crime字段，将其转换为标准化的list格式
#
#     处理逻辑：
#     1. 如果已经是list类型，直接返回
#     2. 如果是字符串且包含标准的list格式，尝试提取并转换为list
#     3. 如果字符串中包含罪名关键词，抽取符合的罪名
#     4. 其他情况返回空list
#
#     参数:
#         text: 输入的crime字段内容
#
#     返回:
#         list: 清洗后的罪名列表
#     """
#     # 1. 如果已经是list类型，直接返回
#     if isinstance(text, list):
#         return text
#
#     # 2. 如果是字符串类型
#     if isinstance(text, str):
#         # 去除首尾空格
#         cleaned_text = text.strip()
#
#         # 情况A: 字符串已经是标准的Python list格式
#         # 例如: "['抢劫罪', '聚众冲击国家机关罪', '容留他人吸毒罪']"
#         try:
#             # 使用ast.literal_eval安全地评估字符串
#             result = ast.literal_eval(cleaned_text)
#             if isinstance(result, list):
#                 # 确保list中的所有元素都是字符串
#                 return [str(item).strip() for item in result if str(item).strip()]
#         except (ValueError, SyntaxError):
#             # 如果不是标准的list格式，继续尝试其他方法
#             pass
#
#         # 情况B: 字符串包含Python代码块标记
#         # 例如: "```python\n['受贿罪', '交通肇事罪（追缴犯罪所得）']\n```"
#         if '```python' in cleaned_text or '```' in cleaned_text:
#             # 提取代码块中的内容
#             code_match = re.search(r'```python\s*(.*?)\s*```', cleaned_text, re.DOTALL)
#             if code_match:
#                 code_content = code_match.group(1).strip()
#                 # 尝试从代码块中提取list
#                 list_match = re.search(r'\[(.*?)\]', code_content, re.DOTALL)
#                 if list_match:
#                     list_content = list_match.group(0)
#                     try:
#                         result = ast.literal_eval(list_content)
#                         if isinstance(result, list):
#                             return [str(item).strip() for item in result if str(item).strip()]
#                     except:
#                         pass
#
#         # 情况C: 字符串包含方括号但格式不标准
#         # 例如: "[受贿罪', '交通肇事罪（追缴犯罪所得）', '滥用职权罪（没收、\n退赔及上缴国库）', '国有企业人员滥用职权罪]"
#         if '[' in cleaned_text and ']' in cleaned_text:
#             # 尝试提取方括号中的内容
#             bracket_match = re.search(r'\[(.*?)\]', cleaned_text, re.DOTALL)
#             if bracket_match:
#                 bracket_content = bracket_match.group(1).strip()
#
#                 # 处理可能的分隔符：逗号、换行符
#                 # 移除换行符和多余空格
#                 bracket_content = re.sub(r'\s+', ' ', bracket_content)
#
#                 # 分割元素
#                 items = []
#                 # 使用正则表达式匹配被单引号或双引号包围的内容
#                 quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", bracket_content)
#                 if quoted_items:
#                     items = quoted_items
#                 else:
#                     # 如果没有引号，按逗号分割
#                     items = [item.strip() for item in bracket_content.split(',') if item.strip()]
#
#                 # 清理每个元素
#                 cleaned_items = []
#                 for item in items:
#                     # 移除可能的尾随标点
#                     item = re.sub(r'^[,\s]+|[,\s]+$', '', item)
#                     # 移除中文括号和内容
#                     item = re.sub(r'（[^）]*）', '', item)
#                     item = re.sub(r'\([^)]*\)', '', item)
#                     if item:
#                         cleaned_items.append(item.strip())
#
#                 if cleaned_items:
#                     return cleaned_items
#
#         # 情况D: 字符串中包含罪名关键词（简单匹配）
#         crime_keywords = ['罪']
#         if any(keyword in cleaned_text for keyword in crime_keywords):
#             # 尝试匹配"XXX罪"格式的内容
#             crimes_found = re.findall(r'([\u4e00-\u9fa5]+罪)', cleaned_text)
#             if crimes_found:
#                 return list(set(crimes_found))  # 去重
#
#         # 情况E: 字符串是单个罪名
#         # 例如: "妨害公务罪"
#         if cleaned_text.endswith('罪'):
#             return [cleaned_text]
#
#     # 3. 其他情况返回空list
#     return []

# def cleanFinishedCrimeAttr():
#     for ridx in all_ridx:
#         cid_list = get_all_cid(ridx)
#         ridx_dir = os.path.join(candidate_path, str(ridx))
#         for cid in cid_list:
#             c_dict = getCandidateDict(ridx, cid)
#             try:
#                 crimes = c_dict['crime']
#             except:
#
#             # clean extract data
#             cleaned_data = cleanCandidateCrimes(crimes)
#             c_dict['crime'] = cleaned_data
#             # save json
#             cid_json_path = os.path.join(ridx_dir, f"{cid}.json")
#             with open(cid_json_path, 'w', encoding='utf-8') as f:
#                 json.dump(c_dict, f, ensure_ascii=False, indent=2)
#             print(f"✅ cid={cid} (ridx={ridx})  {crimes} 已保存至: {cid_json_path}")
#
#
#
# # cleanFinishedCrimeAttr()




