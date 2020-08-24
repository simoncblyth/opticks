#!/usr/bin/env python
import os, numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt

if not 'TMP' in os.environ:
    os.environ['TMP'] = os.path.join("/tmp",os.environ["USER"],"opticks")
pass

name = os.path.basename(os.getcwd()) 
tmpdir = os.path.expandvars("$TMP/%s" % name)
np_load_ = lambda n:np.load(os.path.join(tmpdir,"%s.npy" % n)) 

i = np_load_("inp")
o = np_load_("out")
d = np_load_("dbg")
p = np_load_("pos")

print(name)
print("i %s " % str(i.shape))
print("o %s " % str(o.shape))
print("d %s " % str(d.shape))
print("p %s " % str(p.shape))

plt.ion()


b = d[:,:,:3]  # get rid of the alpha which is always 1.
s = np.where( b[:,:,0] != 0.)     # select non-zero in red channel
c = b[s]      # extract all non-zero result  
nn = np.sum(c*c, axis=1)   # check normalization  
np.allclose( nn, 1.)   

q = p[s]
qq = np.sum(q*q, axis=1)   # check normalization  
np.allclose( qq, 1.5)   


theta = np.arccos(c[:,2])
f_theta = theta/np.pi    # 0->1 
phi = np.arctan2( c[:,1], c[:,0] ) 
f_phi = phi/(2.*np.pi)


imshow = True
if imshow:
    fig, axs = plt.subplots(3)
    fig.suptitle('%s' % tmpdir)
    axs[0].imshow(i[0])
    axs[1].imshow(o, origin='lower')
    axs[2].imshow(d, origin='lower')
    plt.show()                     
pass

hist_xyz = False
if hist_xyz:
    fig2, axs2 = plt.subplots(3)
    axs2[0].hist(c[:,0], bins=100)
    axs2[1].hist(c[:,1], bins=100)
    axs2[2].hist(c[:,2], bins=100)
pass

scatter_norm = False 
if scatter_norm:
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(c[:,0], c[:,1], c[:,2], s=0.1)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.show()
pass

scatter_pos = False
if scatter_pos:
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.scatter(q[:,0], q[:,1], q[:,2], s=0.1)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    plt.show()
pass





