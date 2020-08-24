


import numpy as np
import d6tflow
#import configparser
#cfg = configparser.ConfigParser()
import tasks
task = tasks.GenData()
d6tflow.run(task,forced_all_upstream=True,confirm=False)
import lightgbm

df = tasks.GenData().outputLoad()

df.reset_index(level=0, inplace=True)
import random
randomList = random.sample(range(0,999),250)
filter_mask = df['level_0'].isin(randomList)
df_test = df.loc[filter_mask]
df_train = df.loc[~filter_mask]


colX = ['price', 'dist']
colY = 'score'
df_trainX = df_train[colX]
df_trainY = df_train[colY]
df_testX = df_test[colX]
df_testY = df_test[colY]

m_lgbm = lightgbm.LGBMRanker(random_state=0)
#initial code:
#m_lgbm.fit(df_trainX,df_trainY,group=df_train['uid'].factorize()[0])
#new code:
group1 = np.full([750],5)
m_lgbm.fit(df_trainX,df_trainY, group=group1)
#group is assigned with length of each query
lightgbm.plot_importance(m_lgbm)

#calculate nDCG
df_predictY = m_lgbm.predict(df_testX)

from ndcg import ndcg_score
df_testY = df_testY.to_numpy(dtype = "int64")
result = ndcg_score(df_testY, df_predictY, group = np.full([250],5))
print(result)





