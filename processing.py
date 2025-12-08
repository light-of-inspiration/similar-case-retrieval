# processing.py

import os
import json
import jieba

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
