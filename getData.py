'''
Download dataset locally if does not exist
'''
import os
if os.path.isfile('data_swe.pickle'):
    pass
else:
    try:
        os.system('wget https://zenodo.org/records/14803077/files/data_swe.pickle')
    except:
        raise ImportError('wget is used to download the dataset. ' +
                          'Please download dataset from url manually (see getData.py)')
