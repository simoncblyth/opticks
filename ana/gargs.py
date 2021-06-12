#!/usr/bin/env python

import os, sys, argparse, logging
import numpy as np, math 
from collections import OrderedDict as odict 

log = logging.getLogger(__name__)
sys.path.insert(0, os.path.expanduser("~"))  # assumes $HOME/opticks 

slist_ = lambda s:list(map(str.strip,filter(None,s.split("\n"))))

class GArgs(argparse.Namespace):
    @classmethod
    def Make(cls, args):
        """
        Didnt manage to inherit from Namespace in a way that could copy 
        so workaround with this
        """
        gargs = cls()
        for k,v in vars(args).items():
            setattr(gargs, k, v) 
        pass  
        return gargs

    def figpath(self, name):
        path = os.path.join(self.figdir, "%s%s.png" % (self.figpfx,name) )
        fdir = os.path.dirname(path) 
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        pass 
        return path

    @classmethod
    def volname(cls, idx=0, pfx0="NNVTMCPPMT", pfx1="_PMT_20inch"):  
        """
        """
        dlv = odict()

        dlv[0] = "lMaskVirtual"   
        dlv[1] = "lMask"         
        dlv[2] = "_log" 
        dlv[3] = "_body_log" 
        dlv[4] = "_inner1_log"
        dlv[5] = "_inner2_log" 
        dlv[6] = "lInnerWater"

        return "%s%s%s" % (pfx0, pfx1, dlv[idx]) 

    @classmethod
    def lvname(cls, idx):
        """
        When names change yet again, use to update these:: 

            print("\n".join(g.volumes.keys()))     

        """
        lvxs = slist_(r"""
        PMT_3inch_log
        NNVTMCPPMT_log 
        HamamatsuR12860_log
        """)

        old_lvxs = slist_(r"""
        PMT_3inch_log
        NNVTMCPPMTlMaskVirtual
        HamamatsuR12860lMaskVirtual
        mask_PMT_20inch_vetolMaskVirtual
        NNVTMCPPMT_PMT_20inch_log
        HamamatsuR12860_PMT_20inch_body_log
        """)
        return lvxs[idx]

    @classmethod
    def label(cls, idx):
        labels = slist_(r"""
        tds_ngt
        tds_ngt_pcnk
        tds_ngt_pcnk_sycg
        origin_CGDMLKludge
        """)
        return labels[idx] 

    @classmethod
    def gdmlpath(cls, idx):
        label = cls.label(idx) if type(idx) is int else idx
        return "$OPTICKS_PREFIX/%s.gdml" % label 

    @classmethod
    def parse(cls, doc):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "--path", default="$OPTICKS_PREFIX/tds.gdml")

        defaults = {}
        defaults["lvx"] = cls.volname(2)
        defaults["maxdepth"] = -1    
        defaults["xlim"] = "-330,330"  # 660
        defaults["ylim"] = "-460,200"
        defaults["size"] = "10,8"
        defaults["color"] = "r,g,b,c,y,m,k" 
        defaults["figdir"] = "/tmp/fig"       
        defaults["figpfx"] = "PolyconeNeck"       

        parser.add_argument( "--lvx", default=defaults["lvx"], help="LV name prefix" )
        parser.add_argument( "--maxdepth", type=int, default=defaults["maxdepth"], help="Maximum local depth of volumes to plot, 0 for just root, -1 for no limit" )
        parser.add_argument( "--xlim", default=defaults["xlim"], help="x limits : comma delimited string of two values" )
        parser.add_argument( "--ylim", default=defaults["ylim"], help="y limits : comma delimited string of two values" )
        parser.add_argument( "--size", default=defaults["size"], help="figure size in inches : comma delimited string of two values" )
        parser.add_argument( "--color", default=defaults["color"], help="comma delimited string of color strings" )
        parser.add_argument( "--figdir", default=defaults["figdir"], help="directory path in which to save PNG figures" )
        parser.add_argument( "--figpfx", default=defaults["figpfx"], help="prefix for PNG filename" )

        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=fmt)

        fsplit_ = lambda s:map(float,s.split(",")) 
        #args.xlim = fsplit_(args.xlim)
        #args.ylim = fsplit_(args.ylim)
        #args.size = fsplit_(args.size)
        args.color = args.color.split(",")

        pix = np.array([1280.,720.])
        dpi = 100.
        args.size = pix/dpi

        ylim = np.array([-470,210])
        hy = (ylim[1]-ylim[0])/2
        xlim = np.array([-hy,hy])

        args.xlim = xlim 
        args.ylim = ylim 

        args.suptitle_fontsize = 25 

        return cls.Make(args)


if __name__ == '__main__':
    args = GArgs.parse(__doc__)
    print(args)




 
