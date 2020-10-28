import os.path as osp
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset

# generic sofa dataset
class AnthropometricsDataset(Dataset):
    cols_X = [
        'head width',
        'head height',
        'head depth',
        'pinna offset down',
        'pinna offset back',
        'neck width',
        'neck height',
        'neck depth',
        'torso top width',
        'torso top height',
        'torso top depth',
        'shoulder width',
        'head offset forward',
        'height',
        'seated height',
        'head circumference',
        'shoulder circumference']
    cols_D = [
        'cavum concha height',
        'cymba concha height',
        'cavum concha width',
        'fossa height',
        'pinna height',
        'pinna width',
        'intertragal incisure width',
        'cavum concha depth']
    cols_T = [
        'pinna rotation angle',
        'pinna flare angle']

    def __init__(self, data_path, dataset_type):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.df = None
        load_data = {
            'viking': None,
            'hutubs': self.load_data_hutubs,
            'cipic': self.load_data_cipic,
            'ari_inear': self.load_data_cipic
        }.get(self.dataset_type)
        if load_data:
            load_data()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.df is None:
            return {}
        if isinstance(idx, int):
            return self.df.iloc[idx].to_dict()
        if isinstance(idx, tuple):
            try:
                return self.df.loc[idx].to_dict()
            except KeyError:
                return {lbl: 0. for lbl in self.df.columns}

    @property
    def features(self):
        return [*self.cols_X, *self.cols_D, *self.cols_T]

    def _get_index(self, subjs):
        if self.dataset_type == 'hutubs':
            subjs = [f'{subj:03}' for subj in subjs]
        elif self.dataset_type == 'cipic':
            subjs = [f'{subj:03}' for subj in subjs]
        elif self.dataset_type == 'ari_inear':
            subjs = [str(subj)[1:] for subj in subjs]
        index_L = pd.MultiIndex.from_tuples([('L', s) for s in subjs], names=('ear', 'subj'))
        index_R = pd.MultiIndex.from_tuples([('R', s) for s in subjs], names=('ear', 'subj'))
        return index_L, index_R

    def load_data_cipic(self):
        # load data
        file_path = osp.join(self.data_path, 'anthro.mat')
        data = sio.loadmat(file_path)
        # index
        index_L, index_R = self._get_index(data['id'][:, 0])
        # dataframes for left
        df_L_X = pd.DataFrame(data['X'], index=index_L, columns=self.cols_X)
        df_L_D = pd.DataFrame(data['D'][:, 0:8], index=index_L, columns=self.cols_D)
        df_L_T = pd.DataFrame(data['theta'][:, 0:2], index=index_L, columns=self.cols_T)
        # dataframes for right
        df_R_X = pd.DataFrame(data['X'], index=index_R, columns=self.cols_X)
        df_R_D = pd.DataFrame(data['D'][:, 8:16], index=index_R, columns=self.cols_D)
        df_R_T = pd.DataFrame(data['theta'][:, 2:4], index=index_R, columns=self.cols_T)
        # merge dataframes
        df_L = pd.concat([df_L_X, df_L_D, df_L_T], axis=1)
        df_R = pd.concat([df_R_X, df_R_D, df_R_T], axis=1)
        self.df = pd.concat([df_L, df_R], axis=0)

    def load_data_hutubs(self):
        # load data
        file_path = osp.join(self.data_path, 'anthro.csv')
        df_temp = pd.read_csv(file_path, index_col='SubjectID')
        # index
        index_L, index_R = self._get_index(df_temp.index)
        # current column labels
        cols_x = [c for c in df_temp.columns if 'x' in c]
        cols_l = cols_x + [c for c in df_temp.columns if 'L_' in c]
        cols_r = cols_x + [c for c in df_temp.columns if 'R_' in c]
        # new column labels
        cols_x_new = [self.cols_X[int(c[1:]) - 1] for c in cols_x]
        cols_new = cols_x_new + self.cols_D + ['cavum concha depth (back)', 'crus of helix depth'] + self.cols_T
        # format left
        df_L = df_temp[cols_l]
        df_L.index = index_L
        df_L.columns = cols_new
        # format right
        df_R = df_temp[cols_r]
        df_R.index = index_R
        df_R.columns = cols_new
        # merge dataframes
        self.df = pd.concat([df_L, df_R], axis=0)
