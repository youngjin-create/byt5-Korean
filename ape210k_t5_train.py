import pandas as pd
from tqdm import tqdm
from sympy import sympify
from hanspell import spell_checker
import ast
import json
import re

def normalizetext(text):
    # Percentage to Fraction
    text = text.replace('x=', '')
    text = re.sub('([0-9]+)[:]([0-9]+)', '(\g<1>/\g<2>)', text)
    text = re.sub('([0-9]+)[(]([0-9]+)[/]([0-9]+)[)]', '(\g<1>*\g<3>+\g<2>)/(\g<3>)', text)
    objj = re.findall('(?!(?:\.)?%)\d+(?:\.\d+)?%', text)
    for s in objj:
        text = text.replace(s, '('+s[:-1]+'/100)')
    text = re.sub('([0-9]+)[(]([0-9]+)[/]([0-9]+)[)]', '(\g<1>*\g<3>+\g<2>)/(\g<3>)', text)
    return text


################################### DATASET prepariation ####################################

file_path = 'test.ape.json.kr' #Loading Test Dataset
dict_json = []
with open(file_path) as f:
    for line in f:
        dict_json.append(ast.literal_eval(line))

df_corpus = pd.DataFrame(dict_json) 
df_corpus = df_corpus.astype({"ans": str})

error_list = []
right_list = []
wrong_list = []
for idx, expr in enumerate(df_corpus["equation"].tolist()):
    try:
        exprr = normalizetext(expr)
        x = sympify(exprr)
        ans = sympify(normalizetext(df_corpus['ans'].iloc[idx]))
        if (x - ans)**2 < 1e-5:
            #right_list.append((idx, exprr, x, ans))
            right_list.append(idx)
        else:
            #wrong_list.append((idx, exprr, x, ans))
            wrong_list.append(idx)
    except Exception as e:
        #error_list.append( (idx, expr[2:]) )
        error_list.append(idx)
        #print(e)
blacklist = wrong_list + error_list
df_corpus = df_corpus.loc[set(df_corpus.index) - set(blacklist)] # Filter Out Errors from the dataset

df_corpus["prefix"] = "Solve"
df_corpus["input_text"] = df_corpus["original_text_kr"]
for idx, txt in enumerate(tqdm(df_corpus["input_text"])):
    df_corpus["input_text"].iloc[idx] = spell_checker.check(txt).checked ## 문법검사 실행
df_corpus["target_text"] = df_corpus["equation"].apply(lambda x : x.replace('x=', ''))
eval_df = df_corpus[["prefix", "input_text", "target_text"]]
#eval_df = pd.concat([df_train4]).astype(str)
eval_df.to_csv("data/eval.tsv", "\t", index=False)



file_path = 'valid.ape.json.kr' #Loading Validation Dataset
dict_json = []
with open(file_path) as f:
    for line in f:
        dict_json.append(ast.literal_eval(line))

df_corpus = pd.DataFrame(dict_json) 
df_corpus = df_corpus.astype({"ans": str})

error_list = []
right_list = []
wrong_list = []
for idx, expr in enumerate(df_corpus["equation"].tolist()):
    try:
        exprr = normalizetext(expr)
        x = sympify(exprr)
        ans = sympify(normalizetext(df_corpus['ans'].iloc[idx]))
        if (x - ans)**2 < 1e-5:
            #right_list.append((idx, exprr, x, ans))
            right_list.append(idx)
        else:
            #wrong_list.append((idx, exprr, x, ans))
            wrong_list.append(idx)
    except Exception as e:
        #error_list.append( (idx, expr[2:]) )
        error_list.append(idx)
        #print(e)
blacklist = wrong_list + error_list
df_corpus = df_corpus.loc[set(df_corpus.index) - set(blacklist)] # Filter Out Errors from the dataset

df_corpus["prefix"] = "Solve"
df_corpus["input_text"] = df_corpus["original_text_kr"]
for idx, txt in enumerate(tqdm(df_corpus["input_text"])):
    df_corpus["input_text"].iloc[idx] = spell_checker.check(txt).checked ## 문법검사 실행
df_corpus["target_text"] = df_corpus["equation"].apply(lambda x : x.replace('x=', ''))
valid_df = df_corpus[["prefix", "input_text", "target_text"]]
valid_df.to_csv("data/valid.tsv", "\t", index=False)



file_path = 'train.ape.json.kr'
dict_json = []
with open(file_path) as f:
    for line in f:
        dict_json.append(ast.literal_eval(line))

file_path = 'train2.ape.json.kr'
with open(file_path) as f:
    for line in f:
        dict_json.append(ast.literal_eval(line))

df_corpus = pd.DataFrame(dict_json) 
df_corpus = df_corpus.astype({"ans": str})

error_list = []
right_list = []
wrong_list = []
for idx, expr in enumerate(df_corpus["equation"].tolist()):
    try:
        exprr = normalizetext(expr)
        x = sympify(exprr)
        ans = sympify(normalizetext(df_corpus['ans'].iloc[idx]))
        if (x - ans)**2 < 1e-5:
            #right_list.append((idx, exprr, x, ans))
            right_list.append(idx)
        else:
            #wrong_list.append((idx, exprr, x, ans))
            wrong_list.append(idx)
    except Exception as e:
        #error_list.append( (idx, expr[2:]) )
        error_list.append(idx)
        #print(e)
blacklist = wrong_list + error_list
df_corpus = df_corpus.loc[set(df_corpus.index) - set(blacklist)] # Filter Out Errors from the dataset

df_corpus["prefix"] = "Solve"
#df_corpus["input_text"] = df_corpus["Body"] + " " + df_corpus["Question"]
df_corpus["input_text"] = df_corpus["original_text_kr"]
for idx, txt in enumerate(tqdm(df_corpus["input_text"])):
    df_corpus["input_text"].iloc[idx] = spell_checker.check(txt).checked
#df_corpus["target_text"] = df_corpus["Type"] + ", " + df_corpus["Equation"] + ", " + df_corpus["Answer"]
df_corpus["target_text"] = df_corpus["equation"].apply(lambda x : x.replace('x=', ''))
train_df = df_corpus[["prefix", "input_text", "target_text"]]
train_df.to_csv("data/train.tsv", "\t", index=False)



# train_df = pd.concat([binary_train_df, multi_train_df, sts_train_df, df_train]).astype(str)
##################################  Training Model #########################################
from simpletransformers.t5 import T5Model

train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

model_args = {
    #"max_seq_length": 196,
    "max_seq_length": 512,
    
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "gradient_accumulation_steps": 40,
    "num_train_epochs": 10,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 20000,
    "evaluate_during_training_verbose": False,

    "learning_rate": 0.0001,

    "use_multiprocessing": False,
    "fp16": False,

    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    #"wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
    #"wandb_project": "T5 mixed tasks - Binary, Trans",
    "wandb_project": "T5 math-solve - ape210k",
    #"warmup_steps": 10000,
}
# model_args = {
#     #"max_seq_length": 196,
#     "max_seq_length": 200,
#     "train_batch_size": 4,
#     "eval_batch_size": 16,
#     "gradient_accumulation_steps": 8,
#     "num_train_epochs": 20,
#     "evaluate_during_training": False,
#     "evaluate_during_training_steps": 15000,
#     "evaluate_during_training_verbose": False,
#
#     "use_multiprocessing": False,
#     "fp16": False,
#
#     "save_steps": -1,
#     "save_eval_checkpoints": False,
#     "save_model_every_epoch": False,
#     "reprocess_input_data": True,
#     "overwrite_output_dir": True,
#     #"wandb_project": "T5 mixed tasks - Binary, Multi-Label, Regression",
#     #"wandb_project": "T5 mixed tasks - Binary, Trans",
#     "wandb_project": "T5 mixed tasks - SVAMP",
# }

model = T5Model("mt5", "google/mt5-large", args=model_args)
model.train_model(train_df, eval_data=eval_df)



