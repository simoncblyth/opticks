#!/usr/bin/env python
"""
propagate_at_multifilm.py
=================================

As everything is happening at the origin it would be 
impossible to visualize everything on top of each other. 
So represent the incoming hemisphere of photons with 
points and vectors on the unit hemi-sphere with mom
directions all pointing at the origin and polarization 
vector either within the plane of incidence (P-polarized)
or perpendicular to the plane of incidence (S-polarized). 



                   .
            .              .            
        .                     .
  
     .                          .

    -------------0---------------



Notice with S-polarized that the polarization vectors Z-component is zero 

::

   EYE=-1,-1,1 LOOK=0,0,0.5 PARA=1 ./QSimTest.sh ana


"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
import pyvista as pv
import matplotlib.pyplot as plt

#FOLD = os.environ["FOLD"]
#TEST = os.environ["TEST"]
GUI = not "NOGUI" in os.environ

'''
In [13]: prd=np.load("prd.npy")

In [14]: prd.shape
Out[14]: (1, 2, 4)
'''

class MultiFilmPropagate():
    def __init__(self, src , mutate_src , pmttype = 0 , boundary = 0):
        self.mutate_src_path = mutate_src;
        self.t0 = Fold.Load(src)
        self.t1 = Fold.Load(mutate_src)
        self.src_photon = self.t0.p
        self.mutate_src_photon = self.t1.p
        self.prd = self.t1.prd
        self.pmttype = pmttype
        self.boundary = boundary
        
        pmtStringName={}
        pmtStringName[0] = "kPMT_NNVT"
        pmtStringName[1] = "kPMT_Hama"
        pmtStringName[2] = "kPMT_NNVT_HiQE"
        
        self.pmtStringName = pmtStringName 
         
        direction = {}
        direction[0] = "glass_to_vacuum"
        direction[1] = "vacuum_to_glass"       

        self.direction = direction
        flag = {}
        flag[1024] = "boundary_reflect"
        flag[2048] = "boundary_refract"
        self.flag = flag

        print(" src_photon.shape = {}  {}".format(self.src_photon.shape,self.src_photon[:100,:,:]))
        print(" mutate_src_photon = {}".format(self.mutate_src_photon.shape))
        print(" prd.shape = {} ".format(self.prd.shape))
 
    def draw_fix_wv(self):
        total_c1_ = self.src_photon[:,1,2] #mom.z
        total_c1 = np.absolute(total_c1_)
        reflect_c1_ = self.src_photon[self.mutate_src_photon[:,3,-1].view(np.uint32) == 1024, 1, 2]
        reflect_c1 = np.absolute(reflect_c1_)
        transmit_c1_ = self.src_photon[self.mutate_src_photon[:,3,-1].view(np.uint32) == 2048, 1, 2]
        transmit_c1 = np.absolute(transmit_c1_)
        print("total_c1 = {} , reflect_c1 = {}".format(total_c1.shape,reflect_c1.shape))
        print("total_c1 = {} , reflect_c1 = {}".format(total_c1.shape,reflect_c1.shape))
        
        num_bins = 200
        bins = np.linspace(0.0, 1.0, num_bins+1)

        count_total,   bin_array,  patch = plt.hist(total_c1 , bins)
        count_reflect , bin_array, patch = plt.hist( reflect_c1 , bins)
        count_transmit , bin_array, patch = plt.hist( transmit_c1 , bins)
        plt.cla()        

        reflect_ratio = count_reflect/count_total
        transmit_ratio = count_transmit/count_total
      
        print("count_reflect.shape = {} bin ={}".format(count_reflect.shape,bins.shape))
        plt.scatter(bins[:-1],reflect_ratio, s=0.5,label="reflect")
        plt.scatter(bins[:-1],transmit_ratio,s=0.5,label="Transmit")
        plt.legend()
        plt.xlabel("cos_theta")
        plt.ylabel("probability")
        title = self.mutate_src_path +"\n" +self.pmtStringName[self.pmttype]+ "/" + self.direction[self.boundary]
        plt.title(title)
        plt.show()        

        
        #save data to file which will be used by other programm
        data=np.zeros((3,num_bins),dtype = float )
        data[0,:] = bins[:-1]
        data[1,:] = reflect_ratio
        data[2,:] = transmit_ratio
  
        tmp_file_name = self.mutate_src_path+"pmttype{}_boundary{}_R_T.npy".format(self.pmttype,self.boundary)
        np.save(tmp_file_name,data)
        print("the data save in {}".format(tmp_file_name))

    def draw_with_pv(self, flag):
        
        '''
        1024 BF reflect
        2048 BR
        '''     
        
        p = self.mutate_src_photon
        p = p[p[:,3,-1].view(np.uint32) == flag , :, :]
        
        num_point = 1000

        prd = self.prd
        lim = slice(0,num_point)
        
        #print( " TEST : %s " % TEST)
        #print( " FOLD : %s " % FOLD)
        print( "p.shape %s " % str(p.shape) )
        print( "prd.shape %s " % str(prd.shape) )
        print(" using lim for plotting %s " % lim )
        
        mom = p[:,1,:3]   # hemisphere of photons all directed at origin 
        pol = p[:,2,:3]   # S or P polarized 
        pos = mom          # illustrative choice of position on unit hemisphere 
        
        normal = prd[:,0,:3]  # saved from qprd 
        #point =  prd[:,1,:3]  # not really position but its all zeros... so will do 
        point = np.array( [0,0,0], dtype=np.float32 )
        
        print("mom\n", mom) 
        print("pol\n", pol) 
        print("pos\n", pos) 
        
        label = "pvplt_polarized"+"_"+self.flag[flag]+"_"+self.mutate_src_path
        pl = pvplt_plotter(label=label)   
        

        pvplt_viewpoint( pl ) 
        pvplt_polarized( pl, pos[lim], mom[lim], pol[lim] )
        

        pos = np.zeros((num_point,3),dtype=float)
        pvplt_lines(     pl, pos[lim], mom[lim] )
        
        
        pvplt_arrows( pl, point, normal )
        FOLD=self.mutate_src_path 
        
        label = "pvplt_polarized"+"_"+self.flag[flag]
        outpath = os.path.join(FOLD, "figs/%s.png" % label )
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        pass
        
        print(" outpath: %s " % outpath ) 
        cp = pl.show(screenshot=outpath) if GUI else None

if __name__ == '__main__':
    src = os.path.expandvars("/tmp/${USER}/opticks/QSimTest/hemisphere_s_polarized/")
    #src = os.path.expandvars("/tmp/${USER}/opticks/QSimTest/hemisphere_p_polarized/")
    #src = os.path.expandvars("/tmp/${USER}/opticks/QSimTest/hemisphere_x_polarized/")
    mutate_src = os.path.expandvars("/tmp/${USER}/opticks/QSimTest/propagate_at_multifilm_s_polarized/") 
    #mutate_src = os.path.expandvars("/tmp/${USER}/opticks/QSimTest/propagate_at_multifilm_p_polarized/") 
    #mutate_src =os.path.expandvars("/tmp/${USER}/opticks/QSimTest/propagate_at_multifilm_x_polarized/") 
    a=MultiFilmPropagate(src,mutate_src, pmttype = 0,boundary = 0)
    #a.draw_fix_wv() 
    a.draw_with_pv(flag=1024) 
