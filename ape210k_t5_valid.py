import numpy as np
import pandas as pd
from sympy import sympify
from simpletransformers.t5 import T5Model
import re

# import json
# from datetime import datetime
# from pprint import pprint
# from statistics import mean
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import accuracy_score, f1_score
# from transformers.data.metrics.squad_metrics import compute_exact, compute_f1  


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

# model_args = {
#     "overwrite_output_dir": True,
#     #"max_seq_length": 196,
#     "max_seq_length": 512,
#     "eval_batch_size": 16,
#     # "num_train_epochs": 1,
#     "use_multiprocessing": False,
#     "num_beams": None,
#     "do_sample": True,
#     #"max_length": 50,
#     "top_k": 50,
#     "top_p": 0.95,
#     "num_return_sequences": 3,
# }

model_args = {
    "overwrite_output_dir": True,
    #"max_seq_length": 196,
    "max_seq_length": 512,
    "eval_batch_size": 16,
    # "num_train_epochs": 1,
    "use_multiprocessing": False,
    "num_beams": None,
    "do_sample": True,
    #"max_length": 50,
    "top_k": 1,
    #"top_p": 0.95,
    "num_return_sequences": 1,
}

# Load the trained model
#model = T5Model("mt5", "outputs", args=model_args)

model = T5Model("mt5", "outputs/best_model/", args=model_args)

# Load the evaluation data
#df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)
df = pd.read_csv("data/valid.tsv", sep="\t").astype(str)
df = df[["prefix", "input_text", "target_text"]]

# Prepare the data for testing
to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
]
# Get the model predictions
preds = model.predict(to_predict)

# Taking only the first prediction
preds = [pred[0] for pred in preds]
df["predicted"] = preds


error_list = []
right_list = []
wrong_list = []
for idx, expr in enumerate(df["predicted"].tolist()):
    try:
        result1 = sympify(normalizetext(expr))
        result2 = sympify(normalizetext(df["target_text"].loc[idx]))
        if (result1 - result2)** 2 < 1e-5:
            right_list.append((idx, result1, result2))
        else:
            wrong_list.append((idx, result1, result2))
    except Exception as e:
        error_list.append(idx)
        print(e)

len(right_list) / len(df) * 100 # 정답률
len(wrong_list) / len(df) * 100 # 오답률
len(error_list) / len(df) * 100 # Error


# 결과 살펴보기
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

df.loc[[x[0] for x in right_list]]
df.loc[[x[0] for x in wrong_list]]
df.loc[[x for x in error_list]]

#df[df['predicted'].str.contains("(70-60)*(70*(10-15))")]

# truth = df["target_text"].tolist()
# tasks = df["prefix"].tolist()

# Saving the predictions if needed
# with open(f"predictions/predictions_{datetime.now()}.txt", "w") as f:
#     for i, text in enumerate(df["input_text"].tolist()):
#         f.write(str(text) + "\n\n")
#         f.write("Truth:\n")
#         f.write(truth[i] + "\n\n")
#         f.write("Prediction:\n")
#         for pred in preds[i]:
#             f.write(str(pred) + "\n")
#         f.write(
#             "________________________________________________________________________________\n"
#         )

# with open(f"results/result_{datetime.now()}.json", "w") as f:
#     json.dump(results_dict, f)

# def f1(truths, preds):
#     return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])
# def exact(truths, preds):
#     return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])
# def pearson_corr(preds, labels):
#     return pearsonr(preds, labels)[0]
# def spearman_corr(preds, labels):
#     return spearmanr(preds, labels)[0]
