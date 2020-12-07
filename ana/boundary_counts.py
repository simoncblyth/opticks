#!/usr/bin/env python3
"""

::

    [blyth@localhost GNodeLib]$ ~/opticks/ana/boundary_counts.py 
     0 :      1 : Galactic///Galactic 
     1 :      2 : Galactic///Rock 
     2 :      1 : Rock///Air 
     3 :    191 : Air///Air 
     4 :      1 : Air///LS 
     5 :      1 : Air///Steel 
     6 :      1 : Air///Tyvek 
     7 :    504 : Air///Aluminium 
     8 :    504 : Aluminium///Adhesive 
     9 :  32256 : Adhesive///TiO2Coating 
    10 :  32256 : TiO2Coating///Scintillator 
    11 :      1 : Rock///Tyvek 
    12 :      1 : Tyvek//VETOTyvekSurface/vetoWater 
    13 :      1 : vetoWater/CDTyvekSurface//Tyvek 
    14 :      1 : Tyvek///Water 
    15 :  20660 : Water///Acrylic 
    16 :      1 : Acrylic///LS 
    17 :     46 : LS///Acrylic 
    18 :      8 : LS///PE_PA 
    19 :  27370 : Water///Steel 
    20 :     56 : Water///PE_PA 
    21 :  43213 : Water///Water 
    22 :  45612 : Water///Pyrex 
    23 :  20012 : Pyrex///Pyrex 
    24 :  12612 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
    25 :  12612 : Pyrex/NNVTMCPPMT_PMT_20inch_mirror_logsurf2/NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum 
    26 :   5000 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
    27 :   5000 : Pyrex/HamamatsuR12860_PMT_20inch_mirror_logsurf2/HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum 
    28 :  25600 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
    29 :  25600 : Pyrex/PMT_3inch_absorb_logsurf2/PMT_3inch_absorb_logsurf1/Vacuum 
    30 :      1 : Water///LS 
    31 :      1 : Water/Steel_surface/Steel_surface/Steel 
    32 :   2400 : vetoWater///Water 
    33 :   2400 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
    34 :   2400 : Pyrex/PMT_20inch_veto_mirror_logsurf2/PMT_20inch_veto_mirror_logsurf1/Vacuum 
    [blyth@localhost GNodeLib]$ 


Loadsa Water///Acrylic only one Acrylic///LS 



"""
import sys, os, numpy as np, logging, argparse
log = logging.getLogger(__name__)

from opticks.ana.blib import BLib
from opticks.ana.key import keydir
KEYDIR = keydir()


if __name__ == '__main__':
   
    blib = BLib()
    names = blib.names().split("\n")

    avi = np.load(os.path.join(KEYDIR,"GNodeLib/all_volume_identity.npy"))

    volbnd0 = avi[:,2] & 0xffff
    b0,n0 = np.unique(volbnd0,return_counts=True) 

    for i in range(len(b0)):
        bidx = b0[i]
        bname = names[bidx]
        print("%2d : %6d : %s " % (bidx,n0[i],bname))
    pass

