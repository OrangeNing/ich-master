import pandas as pd
import os
from configs import config_cnn
from tqdm import tqdm
if __name__ == "__main__":
    # Parameters
    # label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural','any']
    train = pd.read_csv(os.path.join(config_cnn.dataset, 'newtest.csv'),index_col=0)
    train.insert(2,'any',0)
    train.loc[train['healthy']==1,'any']=int(0)
    train.loc[train['healthy'] == 0, 'any'] = int(1)
    train = train.drop(['healthy'],axis=1)
    train.to_csv('newtest.csv', index = False)
    # test = pd.read_csv(os.path.join('../data', 'test_submission.csv'))
    # for index in tqdm(range(len(test.index))):
    #     # print(test['ID'].iloc[index][-3:])
    #     if test['ID'].iloc[index][-3:]=='any':
    #         test['ID'].iloc[index] = test['ID'].iloc[index][:-3] + 'healthy'
    #         if test['Label'].iloc[index] == 0:
    #             test['Label'].iloc[index]=1
    # for row in tqdm(test.itertuples()):
    #     if row[2][-3:]=='any':
    #         row[2] = row[2][:-3]+'_healthy'
    #         if row[3] == 0:
    #             row[3] = 1
    print(train)
    # test.to_csv('mytest_new.csv',index=False)
