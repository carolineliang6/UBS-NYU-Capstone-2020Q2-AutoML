import d6tflow
import luigi
import sklearn, sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
#import cfg

class GenData(d6tflow.tasks.TaskPqPandas):
    nusers = luigi.IntParameter(default=1000)
    nresults = luigi.IntParameter(default=5)
    pct_price = luigi.FloatParameter(default=0.6)
    pct_dict = luigi.FloatParameter(default=0.3)

    def run(self):

        nprice = int(self.nusers*self.pct_price)
        ndist = int(self.nusers*self.pct_dict)
        nrand = self.nusers-nprice-ndist

        uid = [f'up{id}' for id in range(nprice)]
        uid += [f'ud{id}' for id in range(ndist)]
        uid += [f'ur{id}' for id in range(nrand)]

        import numpy as np
        np.random.seed(0)

        df = pd.DataFrame({'uid':uid})
        def apply_fun(dfg):
            dft = pd.DataFrame({'uid':[dfg['uid'].values[0]]*self.nresults,'price':np.random.normal(size=self.nresults),'dist':np.random.normal(size=self.nresults)})
            return dft
        df = df.groupby('uid',as_index=False).apply(apply_fun)
        assert df.shape[0]==self.nusers*self.nresults
        for col in ['price','dist']:
            df[f'rank_{col}'] = df.groupby('uid')[col].rank()

        df['score']=0
        for driver,col in {'up':'rank_price','ud':'rank_dist'}.items():
            idxSel = (df['uid'].str.startswith(driver)) & (df[col]==1)
            df.loc[idxSel,'score']=5
            idxSel = (df['uid'].str.startswith(driver)) & (df[col].isin([2,3]))
            df.loc[idxSel,'score']=1
        df['is_book']=df['score']==5
        df['is_click']=df['score']==1

        self.save(df)

