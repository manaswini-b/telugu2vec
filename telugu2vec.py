# coding: utf-8

# Telugu2Vec Language Modeling
# The goal of this is to train Telugu word embeddings using the [fast.ai](http://www.fast.ai/) version of [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182)--basically LSTM with dropouts--
#with data from [Wikipedia](https://archive.org/download/tewiki-20170301/tewiki-20170301-pages-articles.xml.bz2).

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

import dill as pickle
import json
from IPython.display import Image
from IPython.core.display import HTML

from spacy.lang.hi import *

import re
import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.learner import *
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import random
from irtokz import IndicTokenizer

PATH='/home/ubuntu/data/'
EXT_PATH = 'textract/'
TRN_PATH = 'ttrain/'
VAL_PATH = 'tvalid/'
SAMPLE_PATH = 'tsample/'

EXT = f'{PATH}{EXT_PATH}'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'
SAMPLE = f'{PATH}{SAMPLE_PATH}'

ext1 = get_ipython().getoutput('ls {EXT}')
temp = f'{PATH}{EXT_PATH}{ext1[0]}/'
ext_files = get_ipython().getoutput('ls {temp} ')
sample_files = get_ipython().getoutput('ls {VAL}')

get_ipython().run_cell_magic('time', '', '%%prun\n\n\ndef clean_files(extracted_filelist, TRN):    \n    cleaned_all = []\n    for ext_file in extracted_filelist:\n        input_file = f\'{temp}{ext_file}\'\n        with open(input_file,\'r\', encoding=\'utf-8\') as f:\n            raw_txt = f.readlines()\n            cleaned_doc = []\n            for line in raw_txt:\n                new_line = re.sub(\'<[^<]+?>\', \'\', line)\n                new_line = re.sub(\'__[^<]+?__\', \'\', new_line) \n                new_line = new_line.strip()\n                if new_line != \'\':\n                    cleaned_doc.append(new_line)\n\n            new_doc = "\\n".join(cleaned_doc)\n            cleaned_all.append(new_doc)\n            with open(f"{TRN}{ext_file}.txt", "w", encoding=\'utf-8\') as text_file:\n                text_file.write(new_doc)\n    return cleaned_all\n\ncleaned_all = clean_files(ext_files, TRN)')
print(f'Preview:\n{cleaned_all[0][:500]}\n\nLength of list (should be equal to number of documents): {len(cleaned_all)}')

random.shuffle(trn_files)
len_valid = int(0.2 * len(trn_files)) 
val_files = trn_files[:len_valid]
trn_files = trn_files[len_valid:]

trn_files = get_ipython().getoutput('ls {TRN}')
val_files = get_ipython().getoutput('ls {VAL}')
print(trn_files), print(val_files), print(len(trn_files)), print(len(val_files))

def word_tokenize(document):
    nlp = Telugu()
    return [token.text for token in nlp(document)]
    
def docs_tokenize(documents_as_lists):   
    for document in documents_as_lists:
        tokens = word_tokenize(document)
        tokens_list.extend(tokens)
    
    return tokens_list

get_ipython().run_cell_magic('time', '', 'tokens_filename = "tokens_list_telugu.txt"\ntokens_list = []\n\ntry:\n    print(f\'Reading from {tokens_filename}\')\n    with open(tokens_filename, "r") as f:\n         tokens_list = json.load(f)\n    \nexcept FileNotFoundError:\n    print(f\'FileNotFound. Trying to tokenize from cleaned_all\')\n    tokens_list = docs_tokenize(cleaned_all)\n    \n    with open(\'tokens_list_telugu.txt\', \'w\') as outfile:\n        json.dump(tokens_list, outfile)\n\nprint(f\'Found {len(tokens_list)} tokens\')')

assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled

TEXT = data.Field(lower=True, tokenize=word_tokenize)
bs=32   #batch size
bptt=70    #backprop through time

#FILES = dict(train=f'{SAMPLE_PATH}', validation=f'{SAMPLE_PATH}', test=f'{SAMPLE_PATH}')
FILES = dict(train=f'{TRN_PATH}', validation=f'{VAL_PATH}', test=f'{VAL_PATH}')
get_ipython().run_cell_magic('time', '', 'md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=50)')
get_ipython().run_cell_magic('time', '', "pickle.dump(TEXT, open(f'{PATH}/tmodels/TEXT_min_freq50.pkl','wb'))")
len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)

TEXT.vocab.itos[:12]
TEXT.vocab.stoi['మరియు']
md.trn_ds[0].text[:12]
next(iter(md.trn_dl))

txt = md.trn_ds[0].text[:10]
TEXT.numericalize([txt])

em_sz = 300  # size of each embedding vector
nh = 500     # number of hidden activations per layer
nl = 3       # number of layers

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
learner = md.get_model(opt_fn, em_sz, nh, nl,
               dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3

#find suitable learning rates
learner.lr_find(1e-07,1e2)
learner.sched.plot()

learner.fit(1e-3, 1, wds=1e-6, cycle_len=1, cycle_save_name='telugu_adam3_20')
learner.save_encoder('telugu_adam3_20')
get_ipython().system('pip install git+git://github.com/irshadbhat/indic-tokenizer.git')
tok = IndicTokenizer(lang='tel', split_sen=True)

m=learner.model
ss="మొదటి పేజీ గుంటూరు జిల్లా గుంటూరు జిల్లా గుంటూరు జిల్లా వివరము సేకరించిన తేది 2012-01-01 11,391 చ.కి.మీ. ల విస్తీర్ణములో వ్యాపించి, 48,89,230 (2011 గణన) జనాభా కలిగిఉన్నది. ఆగ్నేయాన బంగాళాఖాతము, దక్షిణాన ప్రకాశం జిల్లా, పశ్చిమాన మహబూబ్ నగర్ జిల్లా, మరియు వాయువ్యాన నల్గొండ జిల్లా సరిహద్దులుగా ఉన్నాయి. దీని ముఖ్యపట్టణం గుంటూరుఈ జిల్లాకు అతి పురాతన చరిత్ర వుంది. మౌర్యులు, శాతవాహనులు, పల్లవులు, చాళుక్యులు, కాకతీయులు, విజయనగర రాజులు పరిపాలించారు. పల్నాటి యుద్ధం ఇక్కడే జరిగింది. మొగలు సామ్రాజ్యం, నిజా"
s = [tok.tokenize(ss)]
t=TEXT.numericalize(s)
' '.join(s[0])
m[0].bs=1
m.eval()
m.reset()
res, *other_things = m(t)
m[0].bs=bs

nexts = torch.topk(res[-1], 10)[1]
[TEXT.vocab.itos[o] for o in to_np(nexts)]

print(ss,"\n")
for i in range(100):
    n=res[-1].topk(2)[1]
    n = n[1] if n.data[0]==0 else n[0]
    print(TEXT.vocab.itos[n.data[0]], end=' ')
    res,*_ = m(n[0].unsqueeze(0))
print('...')

m = learner.model
TEXT = pickle.load(open(f'{PATH}tmodels/TEXT_min_freq50.pkl','rb'))
m[0].bs=1
m.eval()

def gen_text(ss,topk):
    s = [word_tokenize(ss)]
    t = TEXT.numericalize(s)
    m.reset()
    pred,*_ = m(t)
    pred_i = torch.topk(pred[-1], topk)[1]
    return [TEXT.vocab.itos[o] for o in to_np(pred_i)]

def gen_sentences(ss,nb_words):
    result = []
    s = [word_tokenize(ss)]
    t = TEXT.numericalize(s)
    m.reset()
    pred,*_ = m(t)
    for i in range(nb_words):
        pred_i = pred[-1].topk(2)[1]
        pred_i = pred_i[1] if pred_i.data[0] < 2 else pred_i[0]
        result.append(TEXT.vocab.itos[pred_i.data[0]])
        pred,*_ = m(pred_i[0].unsqueeze(0))
    return(result)

ss="""మొదటి"""
gen_text(ss,10)
''.join(gen_sentences(ss,50))
emb_weights = list(learner.model.named_parameters())[0][1]
emb_np = to_np(emb_weights.data)
TEXT = pickle.load(open(f'{PATH}/tmodels/TEXT_min_freq50.pkl','rb'))
TEXT.vocab.set_vectors(vectors=emb_weights.data,dim=300,stoi=TEXT.vocab.stoi)
pickle.dump(TEXT, open(f'{PATH}tmodels/TEXT_vec.pkl','wb'))
TEXT_vec = pickle.load(open(f'{PATH}tmodels/TEXT_vec.pkl','rb'))
telugu2vec = pd.DataFrame(to_np(TEXT_vec.vocab.vectors))
telugu2vec.index = TEXT_vec.vocab.itos
telugu2vec.head(10)
telugu2save = telugu2vec[~telugu2vec.index.str.contains(' ')]
#remove lines with weird characters due to bad segmentation
telugu2save.to_csv(f'{PATH}tmodels/telugu2vec.vec',sep=' ',header=False, line_terminator='\n')


