import json
import tensorflow as tf
from d2l import torch as d2l
import thulac

file_path = "../data/translation2019zh_valid.json"
f = open(file_path, "r", encoding="utf-8")
lines = f.readlines()

def preprocess_dataset(lines):
    """预处理英文-中文数据集"""
    text = ""
    C_pun = u'，。！？【】（）《》“‘：；［］｛｝&，．？（）＼％－＋￣~＄#＠=＿、／'
    E_pun = u',.!?[]()<>"\':;[]{}&,.?()\\%-+~~$#@=_//'
    rule = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}  # 将中文标点与英文全角标点转英文半角
    for line in lines:
        if len(line.strip()) == 0:
            continue  # 忽略空行
        line = json.loads(line)
        text += line["english"] + "\n"
        text += line["chinese"].translate(rule) + "\n"

    def continue_space(char, prev_char):
        return char == ' ' and prev_char == ' '

    # 将大写字母转换为小写，使用空格替换不间断空格
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 去除连续空格
    out = ['' if i>0 and continue_space(char, text[i-1]) else char for i, char in enumerate(text)]
    text = ''.join(out)
    chinese, english = [], []
    for i, t in enumerate(text.split("\n")):
        if i % 2 == 0:
            english.append(t)
        else:
            chinese.append(t)
    english.pop()
    return english, chinese

english, chinese = preprocess_dataset(lines)
for i in range(3):
    print(english[i], chinese[i])


def tokenize(english, chinese, num_examples = None):
    """词元化"""
    source = [e.split() for e in english]
    thu = thulac.thulac()
    target = [thu.cut(c)[0] for c in chinese]
    return source, target

source, target = tokenize(english, chinese)
for i in range(3):
    print(source[i], target[i])


