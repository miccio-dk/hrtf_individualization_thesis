import os
import requests
from glob import glob
from tqdm import trange

# download a dataset
def download_sofa(dataset_path, ds_url_file, expected_count):
    file_list = glob(os.path.join(dataset_path, '*.sofa'))
    if len(file_list) == expected_count:
        return len(file_list)
    ds_url = 'http://sofacoustics.org/data/database'
    os.makedirs(dataset_path, exist_ok=True)
    file_count = 0
    for i in trange(166):
        file_url = f'{ds_url}/{ds_url_file}'.format(i)
        file_path = f'{dataset_path}/subj_{i:03}.sofa'
        # check if already exists
        if os.path.exists(file_path):
            pass
        # download file
        r = requests.get(file_url)
        # if download is successful, store
        if r.status_code == 200:
            with open(file_path, 'wb') as fp:
                fp.write(r.content)
            file_count += 1
    return file_count
