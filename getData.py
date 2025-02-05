'''
Download dataset locally if does not exist
'''
import os
if os.path.isfile('data_swe.pickle'):
    pass
else:
    try:
        os.system('wget https://folk.ntnu.no/jamesij/1d/data_swe.pickle')
    except:
        raise ImportError('wget is used to download the dataset. ' +
                          'Please download dataset from url manually (see getData.py)')
