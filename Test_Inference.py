from Attention_Classification.attention_config import attention_config as config
from inference import NewsClassification

import pandas as pd 
from os.path import join
news = NewsClassification(config)

path = 'data/test.csv'
df = pd.read_csv(path)
print('Shape', df.shape)

documents = df['text'].tolist()
results, attn_scores = news.predict(documents)

print('Results Length', len(results))
dd = pd.DataFrame(results, columns=['Document', 'pred'])
dd['true'] = df['labels']
dd.to_csv(join(config['results_dir'], 'test_res.csv'))

df_scores =  pd.DataFrame(attn_scores, columns=['Ids', 'Word_scores', 'Sent_scores'])
df_scores['pred'] = dd['pred']
df_scores['true'] = dd['true']
df_scores.to_csv(join(config['results_dir'], 'scores_test.csv'))

from sklearn.metrics import classification_report
y_true = dd['true'].tolist()
y_pred = dd['pred'].tolist()

classes = news.labels_set

y_true = [classes.index(each) for each in y_true]
y_pred = [classes.index(each) for each in y_pred]

print(classification_report(y_true, y_pred, target_names=classes))