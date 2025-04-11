#!/usr/bin/env python
"""
sreport.py
======================

::

    PLOT=Substamp_ONE_Delta PICK=A ~/o/sreport.sh
    PLOT=Substamp_ONE_Delta PICK=B ~/o/sreport.sh

    PLOT=Substamp_ONE_Etime PICK=A ~/o/sreport.sh
    PLOT=Substamp_ONE_Etime PICK=B ~/o/sreport.sh
    
    PLOT=Substamp_ONE_maxb_scan PICK=A ~/o/sreport.sh
    PLOT=Substamp_ONE_maxb_scan PICK=B ~/o/sreport.sh

    PLOT=Ranges_ONE ~/o/sreport.sh
    PLOT=Ranges_SPAN ~/o/sreport.sh

    PLOT=Substamp_ALL_Etime_vs_Photon ~/o/sreport.sh
    PLOT=Substamp_ALL_Hit_vs_Photon   ~/o/sreport.sh
    PLOT=Substamp_ALL_RATIO_vs_Photon ~/o/sreport.sh

    PLOT=Subprofile_ONE PICK=A ~/o/sreport.sh
    PLOT=Subprofile_ONE PICK=B ~/o/sreport.sh
 
    PLOT=Subprofile_ALL ~/o/sreport.sh
    PLOT=Runprof_ALL    ~/o/sreport.sh    

"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta

def format_large_number(number):
    if number//1000000 == 0:
        if number % 1000 == 0:
            num_K = number//1000
            num = "%dK" % num_K
        else:
            num = "%d" % number
        pass
    elif number//1000000 > 0:
        if number % 1000000 == 0:
            num_M = number//1000000
            num = "%dM" % num_M
        else:
            num = "%d" % number
        pass
    else:
        num = "%d" % number
    pass
    return num



COMMANDLINE = os.environ.get("COMMANDLINE", "")
STEM =  os.environ.get("STEM", "")
HEADLINE = "%s ## %s " % (COMMANDLINE, STEM ) 
JOB =  os.environ.get("JOB", "")
PLOT =  os.environ.get("PLOT", "")
STEM =  os.environ.get("STEM", "")
MODE =  int(os.environ.get("MODE", "2"))
PICK =  os.environ.get("PICK", "AB")
TLIM =  np.array(list(map(int,os.environ.get("TLIM", "0,0").split(","))),dtype=np.int32)


if MODE != 0:
    from opticks.ana.pvplt import * 
pass

labels_ = lambda l:l.view("|S%d" % l.shape[1])[:,0]  
tv_     = lambda a:a.view("datetime64[us]")  

# https://matplotlib.org/stable/gallery/color/named_colors.html
palette = ["red","green", "blue", 
           "cyan", "magenta", "yellow", 
           "tab:orange", "tab:pink", "tab:olive",
           "tab:purple", "tab:grey", "tab:cyan"
           ]


def make_title(meta, method, symbol):
    base = meta.base.replace("/data/blyth/opticks/GEOM/", "")
    smry = meta.smry("GPUMeta,prefix,creator")
    sfmt = meta.smry("stampFmt") 
    titl = "%s:%s %s " % (symbol,method, sfmt) 
    title = " ".join([titl,base,smry]) 
    return title

smry__ = lambda _:NPMeta.Summarize(_)
smry_ = lambda _:list(map(smry__, _))


class Subprofile(object):
    """
    Why VM is so large with CUDA

    * https://forums.developer.nvidia.com/t/high-virtual-memory-consumption-on-linux-for-cuda-programs-is-it-possible-to-avoid-it/67706/4

    """
    FONTSIZE = 20 
    XLABEL = "Time from 1st sprof.h stamp (seconds)"
    YLABEL = "VM and/or RSS sprof.h memory (GB) "

    @classmethod
    def Time_VM_RS(cls, f):
        subprofile = f.subprofile
        assert len(subprofile.shape) == 3
        t0 = subprofile[0,0,0] 
        sp = subprofile.reshape(-1,3)   
        tp = (sp[:,0] - t0)/1e6   ## seconds from start
        vm = sp[:,1]/1e6          ## GB
        rs = sp[:,2]/1e6          ## GB
        return tp, vm, rs 

    @classmethod
    def Title(cls, f):
        meta = f.subprofile_meta
        base = meta.base.replace("/data/blyth/opticks/GEOM/", "")
        smry = meta.smry("GPUMeta,prefix,creator")
        sfmt = meta.smry("stampFmt") 
        title = "\n".join([base,smry,sfmt]) 
        return title


class Subprofile_ALL(object):
    """
    RSS vs time for both "a" and "b"
    """
    def __init__(self, fold, symbol="fold.subprofile"):
        base = eval(symbol)
        label = RUN_META.Title(fold)

        fontsize = Subprofile.FONTSIZE
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
            ax = axs[0]
            for sym in base.ff:
                f = getattr(base, sym)
               
                tp,vm,rs = Subprofile.Time_VM_RS(f)

                if "VM" in os.environ:
                    ax.scatter( tp, vm, label="%s : VM(GB) vs time(s)" % sym.upper())
                    ax.plot(    tp, vm )
                pass
                ax.scatter( tp, rs, label="%s : RSS(GB) vs time(s)" % sym.upper())
                ax.plot( tp, rs )
            pass
            ax.set_xlabel(Subprofile.XLABEL, fontsize=Subprofile.FONTSIZE )
            ax.set_ylabel(Subprofile.YLABEL, fontsize=Subprofile.FONTSIZE )
            ax.legend()
            fig.show()


class Runprof_ALL(object):
    """
    PLOT=Runprof_ALL ~/o/sreport.sh 
    """
    def __init__(self, fold, symbol="fold.runprof"):
        rp = eval(symbol)
        label = "Runprof_ALL " 
        fontsize = Subprofile.FONTSIZE
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label, equal=False)
            ax = axs[0]

            tp = (rp[:,0] - rp[0,0])/1e6
            vm = rp[:,1]/1e6
            rs = rp[:,2]/1e6 

            if "VM" in os.environ:
                ax.scatter( tp, vm, label="%s : VM(GB) vs time(s)" % symbol)
                ax.plot(    tp, vm )
            pass
            ax.scatter( tp, rs, label="%s : RSS(GB) vs time(s)" % symbol )
            ax.plot( tp, rs )
            pass
            ax.set_xlabel(Subprofile.XLABEL, fontsize=Subprofile.FONTSIZE )
            ax.set_ylabel(Subprofile.YLABEL, fontsize=Subprofile.FONTSIZE )

            yl = ax.get_ylim()
            ax.vlines( tp[::4], yl[0], yl[1], color="blue", label="nBeg" ) 
            ax.vlines( tp[1::4], yl[0], yl[1], color="red", label="nEnd" ) 
            ax.vlines( tp[2::4], yl[0], yl[1], color="cyan", label="pBeg" ) 
            ax.vlines( tp[3::4], yl[0], yl[1], color="pink", label="pEnd" ) 

            ax.legend()
            fig.show()
        pass  


class Subprofile_ONE(object):
    """
    PLOT=Subprofile_ONE ~/o/sreport.sh 
    """
    def __init__(self, fold, symbol="fold.subprofile.a"):
        f = eval(symbol)

        meta = f.subprofile_meta
        names = f.subprofile_names
        labels = f.subprofile_labels
        title = RUN_META.Title(fold) 

        tp,vm,rs = Subprofile.Time_VM_RS(f)

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            if "VM" in os.environ:
                ax.scatter( tp, vm, label="VM(GB) vs time(s)")
                ax.plot( tp, vm )
            pass
            ax.scatter( tp, rs, label="RSS(GB) vs time(s)")
            ax.plot( tp, rs )
            ax.set_xlabel(Subprofile.XLABEL, fontsize=Subprofile.FONTSIZE )
            ax.set_ylabel(Subprofile.YLABEL, fontsize=Subprofile.FONTSIZE )
            ax.legend()
            fig.show()
        pass  


class Substamp(object):
    @classmethod
    def ETime(cls, f):
        etime = f.delta_substamp[:,-1]/1e6  
        return etime
    @classmethod
    def Subcount(cls, f, label="photon"):
        _icol = np.where(f.subcount_labels == label)[0] 
        icol = _icol[0] if len(_icol) == 1 else -1 
        subcount = f.subcount[:,icol] if icol > -1 else None
        return subcount
    @classmethod
    def Labels(cls, f):
        labels = f.substamp_labels
        labels_s = smry_(labels)
        return labels_s
    @classmethod
    def DeltaColumn(cls, f, label ):
        _c = np.where( f.substamp_labels == label )[0]
        c = _c[0] if len(_c) == 1 else -1 
        return f.delta_substamp[:,c] if c > -1 else None

    @classmethod
    def Hdr(cls, f):
        labels_s = cls.Labels(f)
        hdr = (" " * 8  + " %4s " * len(labels_s) ) % tuple(labels_s) 
        return hdr
    @classmethod
    def Title(cls, f, symbol="a"):
        meta = f.substamp_meta
        title = make_title(meta, method="Substamp", symbol=symbol)
        return title 


class Substamp_ONE_Etime(object):
    """
    Switches from using subcount_photon to submeta_NumPhotonCollected 
    as that avoids needing to gather and save large photons arrays. 
    """
    def __init__(self, fold, symbol="fold.substamp.a", esym="a" ):

        f = eval(symbol)

        substamp = f.substamp
        meta = f.substamp_meta
        names = f.substamp_names
        delta = f.delta_substamp 

        photon = SUB_META.NumPhotonCollected(fold, esym)/1e6
        etime = Substamp.ETime(f) 

        title = RUN_META.Title(fold)
        #title = Substamp.Title(f, symbol=symbol)
        hdr = Substamp.Hdr(f)

        desc = "\n".join([title, hdr, symbol, repr(delta)])
        print(desc)

        assert len(substamp.shape) == 2 
        assert delta.shape == substamp.shape 
        fontsize = 20 

        deg = 1  # linear   
        linefit = np.poly1d(np.polyfit(photon, etime, deg))
        linefit_label = "line fit:  slope %10.2f    intercept %10.2f " % (linefit.coef[0], linefit.coef[1])

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            ax.scatter( photon, etime, label="etime(s)_vs_photon (millions)")
            ax.plot( photon, linefit(photon), label=linefit_label )

            ax.set_yscale('log')
            ax.set_ylabel("Event time (seconds)", fontsize=fontsize )
            ax.set_xlabel("Number of Photons (Millions)", fontsize=fontsize )
            ax.legend()
            fig.show()
        pass  


class Substamp_ALL_Hit_vs_Photon(object):
    def __init__(self, fold, symbol="fold.substamp", ):
        base = eval(symbol)
        title = RUN_META.Title(fold)
        fontsize = 20 
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            for sym in base.ff:
                f = getattr(base, sym)

                p_symbol = "fold.submeta_NumPhotonCollected.%s[:,0]" % sym 
                #_photon = Substamp.Subcount(f, "photon")
                _photon = eval(p_symbol)

                _hit = Substamp.Subcount(f, "hit")

                photon = _photon/1e6
                hit = _hit/1e6

                deg = 1  # linear   
                linefit = np.poly1d(np.polyfit(photon, hit, deg))
                linefit_label = "%s:line fit:  slope %10.3f    intercept %10.3f " % (sym.upper(), linefit.coef[0], linefit.coef[1])

                ax.scatter( photon, hit, label="%s : hit_vs_photon (millions)" % sym.upper() )
                ax.plot( photon, linefit(photon), linestyle="dotted", label=linefit_label,  )
            pass
            #ax.set_yscale('log')
            ax.set_ylabel("Number of Hits (Millions)", fontsize=fontsize )
            ax.set_xlabel("Number of Photons (Millions)", fontsize=fontsize )
            ax.legend()
            ax.legend()
            fig.show()
        pass


class Substamp_ALL_Etime_vs_Photon(object):
    def __init__(self, fold, symbol="fold.substamp"):
        base = eval(symbol)
        title = RUN_META.Title(fold)
        fontsize = 20 
        YSCALE = os.environ.get("YSCALE", "log")
        YMIN = float(os.environ.get("YMIN", "1e-2"))
        YMAX = float(os.environ.get("YMAX", "1e4"))

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            for sym in base.ff:
                f = getattr(base, sym)

                etime = Substamp.ETime(f) 
                p_symbol = "fold.submeta_NumPhotonCollected.%s[:,0]" % sym 
                _photon = eval(p_symbol)
                photon = _photon/1e6

                deg = 1  # linear   
                linefit = np.poly1d(np.polyfit(photon, etime, deg))
                linefit_label = "%s : line fit:  slope %10.3f    intercept %10.3f " % (sym.upper(), linefit.coef[0], linefit.coef[1])

                ax.scatter( photon, etime, label="%s : etime(s)_vs_photon (millions)" % sym.upper() )
                ax.plot( photon, linefit(photon), label=linefit_label )
            pass
            ax.set_xlim( -5, 105 ); 
            ax.set_ylim( YMIN, YMAX );  # 50*200 = 1e4
            ax.set_yscale(YSCALE)
            ax.set_ylabel("Event time (seconds)", fontsize=fontsize )
            ax.set_xlabel("Number of Photons (Millions)", fontsize=fontsize )
            ax.legend()
            ax.legend()
            fig.show()
        pass


class Substamp_ALL_RATIO_vs_Photon(object):
    """
    PLOT=Substamp_ALL_RATIO_vs_Photon ~/o/sreport.sh
    """
    def __init__(self, fold, symbol="fold.substamp"):
        base = eval(symbol)
        #title = "Substamp_ALL_RATIO_vs_Photon"
        title = RUN_META.Title(fold)

        fontsize = 20 
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            assert "a" in base.ff
            assert "b" in base.ff

            a_photon = fold.submeta_NumPhotonCollected.a[:,0]/1e6
            b_photon = fold.submeta_NumPhotonCollected.b[:,0]/1e6
            assert np.all( a_photon == b_photon )

            a_etime = Substamp.ETime(base.a)
            b_etime = Substamp.ETime(base.b)
            boa_etime = b_etime/a_etime 

            ax.scatter( a_photon, boa_etime, label="boa_etime(s)_vs_photon (millions)"  )
            ax.plot( a_photon, boa_etime, label=None, linestyle="dotted" )  
            pass
            ax.set_ylabel("Geant4/Opticks Event time ratio", fontsize=fontsize )
            ax.set_xlabel("Number of Photons (Millions)", fontsize=fontsize )
            ax.set_ylim( 0, 250)
            ax.set_xlim( -5, 105)
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.legend(loc="lower right")
            fig.show()
        pass


class Substamp_ONE_Delta(object):
    def __init__(self, fold, symbol="fold.substamp.a" ):
        f = eval(symbol)
        delta = f.delta_substamp 
        #title = Substamp.Title(f, symbol=symbol)
        title = RUN_META.Title(fold)

        labels_s = Substamp.Labels(f)
 
        ax = None
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            if TLIM[1] > TLIM[0]:
                ax.set_xlim(*TLIM)
            pass
            for i in range(len(delta)):
                for j in range(len(delta[i])):
                    label = None if i > 0 else labels_s[j]
                    color = palette[j % len(palette)]
                    ax.vlines( delta[i,j], i-0.5, i+0.5, label=label , colors=[color] ) 
                pass
            pass
            ax.legend(loc="center")
            fig.show()
        pass  
        self.ax = ax

class Substamp_ONE_maxb_scan(object):
    def __init__(self, fold, symbol="fold.substamp.a", esym="a" ):
        f = eval(symbol) 
        delta = f.delta_substamp 
        photon = SUB_META.NumPhotonCollected(fold, esym)/1e6
        u_photon = np.unique(photon)
        assert len(u_photon) == 1, u_photon
        n_photon = u_photon[0]
        num = format_large_number(n_subcount_photon)

        _subcount_hit = Substamp.Subcount(f, "hit")   ## TODO : include num hit into metadata

        title0 = Substamp.Title(f, symbol=symbol)
        if PLOT.endswith("_HIT"):
            title1 = "Hit count for %s photons vs MAX_BOUNCE "  % num
        else:
            title1 = "Launch Time[us] for %s photons vs MAX_BOUNCE "  % num
        pass
        title = "\n".join([title0, title1])

        labels_s = Substamp.Labels(f)

        pre = Substamp.DeltaColumn(f, "t_PreLaunch")        
        pos = Substamp.DeltaColumn(f, "t_PostLaunch")        
        launch = pos - pre 
        mxb = np.arange(len(launch))

        sel = slice(18)
        deg = 1  # linear   
        linefit = np.poly1d(np.polyfit(mxb[sel], launch[sel], deg))
        linefit_label = "%s : line fit:  slope %10.2f    intercept %10.2f " % (symbol, linefit.coef[0], linefit.coef[1])

        print(labels_s)
 
        ax = None
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            if PLOT.endswith("_HIT"):
                ax.plot(  mxb, _subcount_hit, label=None )
                ax.scatter(  mxb, _subcount_hit, label="hit count vs max bounce (0->31) ")
                ax.set_ylabel("Hit count", fontsize=20 )
                ax.set_xlabel("OPTICKS_MAX_BOUNCE (aka MAX_TRACE)", fontsize=20 )
            else:
                ax.scatter(  mxb, launch, label="launch time [us] vs max bounce (0->31) ")
                ax.plot( mxb, linefit(mxb), linestyle="dotted", label=linefit_label )
                ax.text(-0.05,  -0.1, HEADLINE, va='bottom', ha='left', family="monospace", fontsize=12, transform=ax.transAxes)
                ax.set_ylabel("GPU Launch Time [us] ", fontsize=12 )
            pass
            ax.legend(loc="lower right")
            fig.show()
        pass  
        self.ax = ax



class Ranges_SPAN(object):
    """
    PLOT=Ranges_SPAN ~/o/sreport.sh 

    TODO: split off QRng upload, probably most of upload_geom from that 

    """
    def __init__(self, fold, symbol="fold"):

        NPC = fold.submeta_NumPhotonCollected.a[:,0]
        #title = "Ranges_SPAN : Total Process Time Accounting : ~20 evt from 0.1 M to 100M photons"
        title = RUN_META.Title(fold)  
        print(title)

        print("NPC:%s" % str(NPC))

        t0 = fold.ranges[0,0]   ## currently ranges only setup for cxs_min.sh 
        ax = None
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            names = ["init", "load_geom", "upload_geom", "simulate", "download", "upload_genstep" ]
            COLORS = ["yellow", "magenta", "cyan",  "red", "green", "blue" ]

            for i, name in enumerate(names):
                color = COLORS[i%len(COLORS)]
                ww = np.where( np.char.endswith( fold.ranges_names, name ))[0]  
                if i > 3:
                    assert len(ww) == len(NPC)
                pass
                for j, w in enumerate(ww): 
                    rg = (fold.ranges[w] - t0)/1e6 
                    ax.axvspan(rg[0], rg[1], alpha=0.5, color=color, label=name if j==0 else None )
                pass
            pass
            ax.legend(loc="center right")
            #ax.set_yscale('log')
            ax.set_xlim(-5,180)
            ax.set_xlabel("cxr_min.sh (CSGOptiXSMTest) Process Run Time[s]", fontsize=20 ); 
            fig.show()
        pass  
        self.ax = ax




class Ranges_ONE(object):
    """
    PLOT=Ranges_ONE ~/o/sreport.sh 
    """
    def __init__(self, fold, symbol="fold"):

        NPC = fold.submeta_NumPhotonCollected.a[:,0]
        print("NPC:%s" % str(NPC))
        title = RUN_META.Title(fold)
        print(title)

        names = ["simulate", "download", "upload_genstep" ]


        COLORS = ["red", "green", "blue" ]
        ax = None
        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]
            for i, name in enumerate(names):
                color = COLORS[i%len(COLORS)]
                w = np.where( np.char.endswith( fold.ranges_names, name ))[0]  
                rgs = fold.ranges[w] 
                assert len(rgs) == len(NPC)
                ax.plot( NPC/1e6, rgs[:,2]/1e6, color=color )  
                ax.scatter( NPC/1e6, rgs[:,2]/1e6, label="%s" % name, color=color )  
            pass
            ax.legend(loc="center right")
            ax.set_yscale('log')
            ax.set_ylim(5e-4,50)
            ax.set_xlim(-5,105)
            ax.set_ylabel("Time (seconds)", fontsize=20); 
            ax.set_xlabel("Number of \"TORCH\" Photons from CD center (Millions)", fontsize=20 ); 
            fig.show()
        pass  
        self.ax = ax




class SUB_META(object):
    @classmethod
    def NumPhotonCollected(cls, fold, esym):
        assert esym in ["a", "b"]
        p_symbol = "fold.submeta_NumPhotonCollected.%s[:,0]" % esym 
        photon = eval(p_symbol)
        return photon


class RUN_META(object):
    @classmethod
    def _QSim__Switches(cls, fold):
        """
        eg: 'CONFIG_Release PRODUCTION WITH_CHILD WITH_CUSTOM4 PLOG_LOCAL '
        """
        SKIPS = "WITH_CHILD PLOG_LOCAL".split(); 
        switches = list(filter(lambda _:not _.startswith("NOT-"), fold.run_meta.QSim__Switches.split(","))) 
        switches = list(filter(lambda _:not _ in SKIPS, switches )) 
        return switches 

    @classmethod
    def QSim__Switches(cls, fold):
        """
        eg: 'CONFIG_Release PRODUCTION WITH_CHILD WITH_CUSTOM4 PLOG_LOCAL '
        """
        switches = cls._QSim__Switches(fold) 
        return " ".join(switches)  

    @classmethod
    def QSim__RNGLabel(cls, fold):
        switches = cls._QSim__Switches(fold) 
        return "RNG_PHILOX" if "RNG_PHILOX" in switches else "RNG_XORWOW"   


    @classmethod
    def ABCD_Title(cls, *rr):
        SCRIPT = None
        for i, r in enumerate(rr):
            _SCRIPT = getattr(r.run_meta, 'SCRIPT', "cxs_min.sh")
            if i == 0:
                SCRIPT = _SCRIPT 
            else:
                assert SCRIPT == _SCRIPT 
            pass 
        pass
        topline = "%s ## %s " % ( COMMANDLINE, SCRIPT )

        cvar = ["RUNNING_MODE","EVENT_MODE","MAX_BOUNCE"] 
        #cvar += ["MAX_PHOTON"]
        #cvar += ["NUM_PHOTON"]

        ctrls = [] 
        for var in cvar:
            val = None
            for i, r in enumerate(rr):
                _val = getattr(r.run_meta, "OPTICKS_%s" % var, "?" )

                if i == 0:
                    val = _val
                else:
                    assert val == _val, ( val,  _val ) 
                pass
            pass
            ctrls.append("%s:%s" % (var,val)) 
        pass
        ctrl = " ".join(ctrls)

        for i, r in enumerate(rr):
            SWITCHES = cls.QSim__Switches(r)
            print(SWITCHES)
        pass  
        title = "\n".join([topline, ctrl])
        return title 

    @classmethod
    def AB_Title(cls, a, b):
        """
        :param a: sreport Fold
        :param b: sreport Fold
        :return str: title 

        In addition to providing the title this also
        asserts consistency between the sreport fold
        for the below run_meta:

        1. SCRIPT
        2. OPTICKS_RUNNING_MODE, OPTICKS_EVENT_MODE, OPTICKS_MAX_BOUNCE, OPTICKS_MAX_PHOTON
        3. QSim__Switches
        """
        a_SCRIPT = getattr(a.run_meta, 'SCRIPT', "cxs_min.sh")  
        b_SCRIPT = getattr(b.run_meta, 'SCRIPT', "cxs_min.sh")  
        assert a_SCRIPT == b_SCRIPT, (a_SCRIPT, b_SCRIPT)
        SCRIPT = a_SCRIPT


        topline = "%s ## %s " % ( COMMANDLINE, SCRIPT )

        cvar = ["RUNNING_MODE","EVENT_MODE","MAX_BOUNCE","MAX_PHOTON"] 
        #cvar += ["NUM_PHOTON"]

        ctrls = [] 
        for var in cvar:
            a_val = getattr(a.run_meta, "OPTICKS_%s" % var, "?" )
            b_val = getattr(a.run_meta, "OPTICKS_%s" % var, "?" )
            assert a_val == b_val, (a_val, b_val)
            val = a_val
            ctrls.append("%s:%s" % (var,val)) 
        pass
        ctrl = " ".join(ctrls)


        a_SWITCHES = cls.QSim__Switches(a)
        b_SWITCHES = cls.QSim__Switches(b)
        assert a_SWITCHES == b_SWITCHES, (a_SWITCHES, b_SWITCHES) ## TODO: layout when not matched
        SWITCHES = a_SWITCHES 

        title = "\n".join([topline, ctrl])
        return title 

    @classmethod
    def GPUMeta(cls, fold, simplify=True):
        gpu = fold.run_meta.GPUMeta
        if simplify and (gpu.startswith("0:") or gpu.startswith("1:")):
            gpu = gpu[2:]
        pass
        return "%s : %s" % (fold.symbol, gpu) 
 

    @classmethod
    def Title(cls, fold):
        SCRIPT = getattr(fold.run_meta, 'SCRIPT', "cxs_min.sh")  
        GPUMeta = fold.run_meta.GPUMeta 
        topline = "%s ## %s : %s " % ( COMMANDLINE, SCRIPT, GPUMeta )
        SWITCHES = cls.QSim__Switches(fold)

        cvar = ["RUNNING_MODE","EVENT_MODE","MAX_BOUNCE","MAX_PHOTON"] 
        #cvar += ["NUM_PHOTON"]

        ctrls = [] 
        for var in cvar:
            val = getattr(fold.run_meta, "OPTICKS_%s" % var, "?" )
            ctrls.append("%s:%s" % (var,val)) 
        pass
        ctrl = " ".join(ctrls)

        title = "\n".join([topline, SWITCHES, ctrl])
        return title 






if __name__ == '__main__':

    print("[sreport.py:PLOT[%s]" % PLOT ) 
    print("[sreport.py:fold = Fold.Load" ) 
    fold = Fold.Load("$SREPORT_FOLD", symbol="fold")
    print("]sreport.py:fold = Fold.Load" ) 

    print("[sreport.py:repr(fold)" ) 
    print(repr(fold))
    print("]sreport.py:repr(fold)" ) 


    SWITCHES = RUN_META.QSim__Switches(fold)
    print("MODE:%d PICK:%s SWITCHES:%s " % (MODE, PICK, SWITCHES) ) 
    print("COMMANDLINE:%s" % COMMANDLINE)

    if PLOT.startswith("Substamp_ONE") and hasattr(fold, "substamp"):
        for e in PICK:
            esym = e.lower()
            f = getattr(fold.substamp, esym, None)
            symbol = "fold.substamp.%s" % esym 
            pass
            if f is None: continue 
            if PLOT.startswith("Substamp_ONE_Delta"): Substamp_ONE_Delta(fold, symbol=symbol )
            if PLOT.startswith("Substamp_ONE_Etime"): Substamp_ONE_Etime(fold, symbol=symbol, esym=esym )
            if PLOT.startswith("Substamp_ONE_maxb_scan"): Substamp_ONE_maxb_scan(fold, symbol=symbol, esym=esym )
        pass
    pass  
    ## Ranges only working for cxs_min.sh : based on SProf.hh run metadata : that needs work for G4CXTest
    if PLOT.startswith("Ranges_ONE") and hasattr(fold, "ranges") and hasattr(fold,"submeta_NumPhotonCollected") :
        Ranges_ONE(fold, symbol="fold")  ## process activity line plots 
    if PLOT.startswith("Ranges_SPAN") and hasattr(fold, "ranges") and hasattr(fold,"submeta_NumPhotonCollected") :
        Ranges_SPAN(fold, symbol="fold")  ## colorfull process activity plot 
    pass
    if PLOT.startswith("Substamp_ALL"): 
        if hasattr(fold, "substamp"):
            if PLOT.startswith("Substamp_ALL_Etime_vs_Photon"): Substamp_ALL_Etime_vs_Photon(  fold, symbol="fold.substamp" )
            if PLOT.startswith("Substamp_ALL_Hit_vs_Photon"): Substamp_ALL_Hit_vs_Photon(      fold, symbol="fold.substamp" )
            if PLOT.startswith("Substamp_ALL_RATIO_vs_Photon"): Substamp_ALL_RATIO_vs_Photon(  fold, symbol="fold.substamp" )
        else:
            print(".sreport.py: fold does not have substamp : CANNOT PLOT [%s]" % PLOT)
        pass
    pass
    if PLOT.startswith("Subprofile_ONE") and hasattr(fold, "subprofile"):
        for e in PICK:
            f = getattr(fold.subprofile, e.lower(), None)
            symbol = "fold.subprofile.%s" % e.lower() 
            if f is None: continue
            Subprofile_ONE(fold, symbol=symbol)   ## RSS vs time : technical 
        pass
    pass
    if PLOT.startswith("Subprofile_ALL") and hasattr(fold, "subprofile"):
        Subprofile_ALL(fold, symbol="fold.subprofile")  ## RSS vs time 
    pass
    if PLOT.startswith("Runprof_ALL") and hasattr(fold, "runprof"):
        Runprof_ALL(fold, symbol="fold.runprof" )  ## RSS vs time : profile plot 
    pass
    print("]sreport.py:PLOT[%s]" % PLOT ) 
pass


