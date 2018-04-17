%reload_ext autoreload
%autoreload 2
%matplotlib inline

import torch
from fastai.imports import *

from fastai.conv_learner import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

sz=224
arch=resnet34
bs=24

PATH = 'data/breedsdog/'
label_csv = f'{PATH}labels.csv'

print (label_csv)

n = len(list(open(label_csv)))-1

val_idxs = get_cv_idxs(n)
val_idxs

label_df = pd.read_csv(label_csv)
label_df.tail()

mytfm = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.05);
data = ImageClassifierData.from_csv(PATH,'train',f'{PATH}/labels.csv',bs, mytfm,test_name='test', val_idxs=val_idxs, suffix='.jpg')

size_dict = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}

row_sz, col_sz = list(zip(*size_dict.values()))

row_sz = np.array(row_sz)
col_sz = np.array(col_sz)

plt.hist(row_sz)
plt.hist(col_sz[col_sz < 1000])

# Next puzzle, try with a different dropoff rate.
lrn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)
lrn.fit(1e-2, 8)

'''
epoch      trn_loss   val_loss   accuracy                   
    0      1.3793     0.741185   0.797456  
    1      0.979428   0.634091   0.815068                     
    2      0.811252   0.535539   0.829746                     
    3      0.726349   0.640568   0.824364                     
    4      0.686643   0.579505   0.831213                     
    5      0.678264   0.579586   0.827299                     
    6      0.571453   0.598132   0.832192                     
    7      0.563118   0.608916   0.832192  
'''

log_preds, y = lrn.TTA()

probs = np.exp(log_preds)

print(np.max(probs))
print(np.min(probs))

log_preds[:4]

probs = np.exp(log_preds)

imr = ImageModelResults(data.val_ds,log_preds)

most_uncertain = np.argsort(np.average(np.abs(probs-(1/num_classes)), axis = 1))[:4]
idxs_col = np.argsort(np.abs(probs[most_uncertain,:]-(1/num_classes)))[:4,-1]
plot_val_with_title(most_uncertain, "Most uncertain predictions", idxs_col)
