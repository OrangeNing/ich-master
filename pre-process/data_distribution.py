
import pandas as pd
import os
import matplotlib.pyplot as plt


from configs import config_cnn as cfg
# path_data = path_data = cfg.dataset
# # train = pd.read_csv(os.path.join(path_data, 'newtrain.csv'))
# # testdf = pd.read_csv(os.path.join(path_data, 'newtest.csv'))
# train = pd.read_csv(os.path.join(path_data, 'newtrain.csv'))
def data_distribution(train):
    train_healthy = train[train['any'] == 0]
    train_epidural = train[train['epidural']==1]
    train_intraparenchymal = train[train['intraparenchymal']== 1]
    train_intraventricular = train[train['intraventricular']==1]
    train_subarachnoid = train[train['subarachnoid']==1]
    train_subdural = train[train['subdural']==1]
    list = []
    list.append(len(train))
    list.append(len(train_healthy))
    list.append(len(train_epidural))
    list.append(len(train_intraparenchymal))
    list.append(len(train_intraventricular))
    list.append(len(train_subarachnoid))
    list.append(len(train_subdural))
    return list
if __name__ == '__main__':
    path_data = 'E:\ICH'
    train = pd.read_csv(os.path.join(path_data, 'newtrain_any.csv'))
    distribution = data_distribution(train)
    name_list = ['ALL','Healthy','EDP','IPH','IVH','SAH','SDH']
    print(distribution)
    # plt.bar(range(len(distribution)),distribution,tick_label = name_list )
    # plt.title('The data distribution of the new testing dataset')
    # plt.show()
