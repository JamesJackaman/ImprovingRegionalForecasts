'''
Download dataset locally if does not exist
'''
import os
if os.path.isfile('data_swe.pickle'):
    pass
else:
    os.system('wget https://folk.ntnu.no/jamesij/1d/data_swe.pickle')
