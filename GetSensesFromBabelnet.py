import babelnet as bn
from babelnet.language import Language
import pandas as pd
import json

def get_word_senses(dic, word):
    if(word in dic):
        return dic
    senses_list = []
    senses = bn.get_senses(word, from_langs=[Language.EN], to_langs=[Language.DE])
    for sense in senses:
        senses_list.append(str(sense.full_lemma).lower())
    dic[word] = senses_list
    return dic

en_dict = {}

f = open("dict.txt", 'a+')
en_profanity = pd.read_csv('data/profanity_en.csv')
en_profanity = en_profanity[['text', 'canonical_form_1', 'canonical_form_2', 'canonical_form_3']]

for index, row in en_profanity.iterrows():
    if(row['text'] not in en_dict):
        senses_list = []
        senses = bn.get_senses(row['text'], from_langs=[Language.EN], to_langs=[Language.DE])
        if(not pd.isnull(row['canonical_form_1'])):
            en_dict = get_word_senses(en_dict, row['canonical_form_1'])
            senses_list += en_dict[row['canonical_form_1']]
        
            if(not pd.isnull(row['canonical_form_2'])):
                en_dict = get_word_senses(en_dict, row['canonical_form_2'])
                senses_list += en_dict[row['canonical_form_2']]

                if(not pd.isnull(row['canonical_form_3'])):
                    en_dict = get_word_senses(en_dict, row['canonical_form_3'])
                    senses_list += en_dict[row['canonical_form_3']]

 
        for sense in senses:
            senses_list.append(str(sense.full_lemma).lower())
    else:
        senses_list = en_dict[row['text']]
    print(row['text'], senses_list, file=f)
    en_dict[row['text']] = senses_list

