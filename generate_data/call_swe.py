import argparse
import pickle
import os

import swe

if __name__=="__main__":
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('iteration', type=int, default=1)
    args, _ = parser.parse_known_args()
    
    generator = swe.generator()
    out_coarse = swe.swe(generator,resolution='coarse')
    out_fine = swe.swe(generator,resolution='fine')

    out = {'coarse': out_coarse,
           'fine': out_fine}

    iteration = args.iteration
    filename = 'tmp/swe%s.pickle' % iteration
    file = open(filename,'wb')
    pickle.dump(out,file)
    file.close()
