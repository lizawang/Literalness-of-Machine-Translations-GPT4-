import pickle
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import jieba
import sys
from unicodedata import category
import copy
import os


def read_sentPairs():
    en_lines, zh_lines = [], []
    with open("opus-en-zh", "r", encoding="utf-8") as f:
        for line in f:
            en, zh = line.strip().split(" ||| ")
            en_lines.append(en)
            zh_lines.append(zh)
    return en_lines, zh_lines

def read_align_indexes():
    align_indexes = []
    with open("en-zh_alignments.i", "r", encoding="utf-8") as f:
        for line in f:
            align_indexes.append(line.split())
    return align_indexes

def get_word_aligns(en_lines, zh_lines, align_indexes):
    # in total 17451546 sentences
    print(len(en_lines), len(zh_lines), len(align_indexes))
    word_aligns_en2zh = []
    for i in range(len(zh_lines)):
        zh_sent = zh_lines[i].split()
        en_sent = en_lines[i].split()
        f_align = align_indexes[i]
        for n in f_align:
            e, z = int(n.split("-")[0]), int(n.split("-")[1])
            en2zh_word_pair = en_sent[e], zh_sent[z]
            word_aligns_en2zh.append(en2zh_word_pair)
    return word_aligns_en2zh

if __name__ == "__main__":
    # en_lines, zh_lines = read_sentPairs()
    # print(en_lines[:3], zh_lines[:3])
    # align_indexes = read_align_indexes()
    # print(align_indexes[:3])
    # word_aligns_en2zh = get_word_aligns(en_lines, zh_lines, align_indexes)
    # print(word_aligns_en2zh[:30])
    # with open("word_aligns_en2zh.pickle", "wb") as f:
    #     pickle.dump(word_aligns_en2zh, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # with open("word_aligns_en2zh.pickle", "rb") as f2:
    #     word_aligns_en2zh = pickle.load(f2)
    # # in total 2098221 word pairs
    # print(len(Counter(word_aligns_en2zh)))
    # d = Counter(word_aligns_en2zh).most_common()
    # print(d[500000])
    # d = d[:500000]
    # en2zh = {}
    # zh2en = {}
    # for pair in d:
    #     en, zh = pair[0][0].lower(), pair[0][1].lower()
    #     count = int(pair[1])
    #     print("pair:", en, zh, count)
    #     if en not in en2zh:
    #         en2zh[en] = [zh]
    #     elif len(en2zh[en]) < 10:
    #         if zh not in en2zh[en]:
    #             en2zh[en].append(zh)
    #     if zh not in zh2en:
    #         zh2en[zh] = [en]
    #     elif len(zh2en) < 10:
    #         if en not in zh2en[zh]:
    #             zh2en[zh].append(en)
    # print(list(zh2en.items())[:3])
    # print(list(en2zh.items())[:3])
    # with open("en-zh_dict.pickle", "wb") as f3:
    #     pickle.dump(en2zh, file=f3, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("zh-en_dict.pickle", "wb") as f4:
    #     pickle.dump(zh2en, file=f4, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("#####", os.path.abspath("../../mt_project/en_zh_scores.csv"))
    data = pd.read_csv(os.path.abspath("../../mt_project/en_zh_scores.csv"), encoding="utf-8")
    #data.drop_duplicates("Source", inplace=True)
    data.sort_values(by=["GPT4_Score"], inplace=True)
    src_sents, tgt_sents, gpt4_scores = data["Source"].to_list(), data["GPT_Output"].to_list(), data["GPT4_Score"].to_list()
    print(src_sents[:2], tgt_sents[:2])
    
    with open(os.path.abspath("../en-zh_dict.pickle"), "rb") as f3:
        en_zh_dict = pickle.load(f3)
    all_src = en_zh_dict.keys()
    ## en-zh dictionary size: 158139
    print("dictionary size:", len(all_src))
    all_tgt = []
    for v in en_zh_dict.values():
        all_tgt += v
    
    punctuation_en =  [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")] + ["..", "...", "....", ".....", "]", "[", "/", "``", "<", ">"]
    punctuation_zh = [p for p in "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.，"] 
    punctuation = punctuation_en + punctuation_zh
    #print(punctuation_en)
    literalness = {}
    found_in_dict = []
    print("###src sents total count:", len(src_sents))
    for i in range(len(src_sents)):
        src, tgt, align = [], [], []
        en_sent, zh_sent = src_sents[i], tgt_sents[i]
        en_sent = "".join([c for c in en_sent if c not in punctuation])
        zh_sent = "".join([c for c in zh_sent if c not in punctuation])
        en_sent = [w.lower() for w in word_tokenize(en_sent.strip()) if w != " "]
        zh_sent = [w.lower() for w in jieba.lcut(zh_sent.strip()) if w != " "]
        zh_sent_copy = copy.deepcopy(zh_sent)
        if i == 1:
            print("sent pair###:", en_sent, zh_sent)
        ## shared English words and numbers
        #shared = [w_e for w_e in en_sent if w_e in zh_sent and w_e not in punctuation_en]
        #align += shared
        for w in en_sent:
            if w not in punctuation:
                if w in all_src:
                    src.append(w)
                    w_tgt = en_zh_dict[w]
                    same = [] 
                    if i == 1:
                        print(w)
                        print("tgt####:", w_tgt)
                        print("z_sent####:", zh_sent_copy) 
                    for t in zh_sent_copy:  
                        if t in w_tgt and t not in punctuation: 
                            same.append(t)               
                    ## one src can only match to one target, the most likely one
                    if len(same):
                        align.append(same[0])
                        ## the same item in the zh_sent cannot be matched twice
                        zh_sent_copy.remove(same[0]) 
        for w in zh_sent:
            if w in all_tgt and w not in punctuation:
                tgt.append(w)
        if i == 1:    
            print("This###:", src, tgt, align)
        if len(src)==0 and len(tgt)==0:
            print(i, en_sent, zh_sent)
            literal_score = 0.0
        else:
            literal_score = 2*len(align)/(len(src)+len(tgt))
        if (" ".join(en_sent), " ".join(zh_sent)) not in literalness:
            literalness[(" ".join(en_sent), " ".join(zh_sent))] = [literal_score]
        else:
            literalness[(" ".join(en_sent), " ".join(zh_sent))].append(literal_score)
        found_in_dict.append((src, tgt, align))
              
    for k, v in literalness.items():
        if len(v) > 1: 
            print(k, v)
    # with open("literalness_ref.tsv", "w", encoding="utf-8") as f5, open("found_in_dict_ref.tsv", "w", encoding="utf-8") as f6:
    #     for k, v in literalness.items():
    #         if len(v) == 1:
    #             out = k[0] + "\t" + k[1] + "\t" + str(v[0])
    #             print(out, file=f5)
    #         elif len(v) > 1:
    #             for s in v:
    #                 out = k[0] + "\t" + k[1] + "\t" + str(s)
    #                 print(out, file=f5)
    #     for s, t, a in found_in_dict:
    #         out_ = "$" + ",".join(s) + "\t" + "#" + ",".join(t) + "\t" + "@" + ",".join(a)
    #         print(out_, file=f6)
        
        
        
 
    
        