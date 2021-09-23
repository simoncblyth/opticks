#!/usr/bin/env python
"""
::

    ipython -i ImageNPYConcatTest.py


"""
import os 
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':

    plt.ion()

    ap = os.path.expandvars("$TMP/SPPMTest_MakeTestImage_old_concat.npy")
    bp = os.path.expandvars("$TMP/SPPMTest_MakeTestImage_new_concat.npy")
    paths = [ap,bp]

    a = np.load(ap)
    b = np.load(bp)
    imgs = [a,b]

    for path,img in zip(paths,imgs):
        print(img.shape)
        assert img.shape[0] < 4       
        layers = img.shape[0]
        fig, axs = plt.subplots(layers, figsize=[12.8, 7.2])
        fig.suptitle("%s %r" % (imgpath,img.shape))
        for i in range(layers): 
            axs[i].imshow(img[i], origin='lower')
        pass
        plt.show()                           
    pass






