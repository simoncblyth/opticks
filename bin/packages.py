#!/usr/bin/env python

import numpy as np
import os, re, textwrap
from collections import OrderedDict as odict
from opticks.ana.rsttable import RSTTable


class Packages(object):
    PATH = "$HOME/opticks/packages.rst"
    LABELS = ["pkg", "hh/cc/cu/py", "desc" ] 
    REMOVED = "cudarap thrustrap optixrap okop oglrap opticksgl opticksgeo cfg4"
    END_OF_LIFE = "boostrap npy optickscore ggeo extg4 CSG_GGeo GeoChain"
    ACTIVE = "CSG CSGOptiX qudarap u4 gdxml g4cx sysrap"

    @classmethod
    def ColorMap(cls):
        cm = odict()
        for key in cls.REMOVED.split(): cm[key] = "r" 
        for key in cls.END_OF_LIFE.split(): cm[key] = "b" 
        for key in cls.ACTIVE.split(): cm[key] = "g" 
        return cm 

    @classmethod
    def Lines(cls):
        return open(os.path.expandvars(cls.PATH),"r").read().splitlines()

    grup_ptn = re.compile("^(.*)\s*$")
    undr_ptn = re.compile("^(-+)\s*$")
    name_ptn = re.compile("^(\S*)\s*$")
    desc_ptn = re.compile("^\s{4}(.*)\s*$") 

    def __init__(self, stat):
        lines = self.Lines()

        apkg = odict()
        gpkg = odict()

        cur_grup = None

        for i in range(len(lines)-1):
            cur = lines[i]
            nex = lines[i+1] 

            grup_match = self.grup_ptn.match(cur)
            undr_match = self.undr_ptn.match(nex)
            grup = grup_match.groups()[0] if not grup_match is None else None
            undr = undr_match.groups()[0] if not undr_match is None else None
            pass
            if not grup is None and not undr is None:  
                #print("grup: [%s] " % grup )
                cur_grup = grup
                gpkg[cur_grup] = odict()  
            pass

            name_match = self.name_ptn.match(cur)
            desc_match = self.desc_ptn.match(nex)
            name = name_match.groups()[0] if not name_match is None else None
            desc = desc_match.groups()[0] if not desc_match is None else None
            pass
            if not name is None and not desc is None:  
                #print("name: [%s] desc:[%s] " % (name, desc))  
                apkg[name] = desc
                if not cur_grup is None:
                    gpkg[cur_grup][name] = desc
                pass
            pass
        pass
        self.lines = lines 
        self.apkg = apkg
        self.gpkg = gpkg
        self.stat = stat


    def repr_apkg(self):
        apkg = self.apkg
        lines = []
        for k,v in apkg.items():
            lines.append("%15s : %s " % (k,v ))
        pass 
        return "\n".join(lines)


    def repr_gpkg(self):
        gpkg = self.gpkg
        lines = []

        nn = list(gpkg.keys()) 
        for i in range(len(nn)):
            n = nn[i]
            if n.startswith("SKIP"): continue 

            lines.append("")
            lines.append("")
            lines.append(n)
            lines.append("")

            g = gpkg[n]
            kk = list(g.keys())
            for j in range(len(kk)):
                k = kk[j]  # pkg name
                v = g[k]   # pkg desc
                st = self.stat.desc(k).split(" ")[1]  # code counts
                lines.append("%15s : %12s : %s" % (k, st, v))
            pass 
        pass 
        return "\n".join(lines)


    def __repr__(self):
        return self.repr_gpkg()
        #return self.repr_apkg()

    def __str__(self):
        return "\n".join(self.lines)

    def group(self, i):
        gpkg = self.gpkg
        nn = list(gpkg.keys()) 
        assert i < len(nn)
        gn = nn[i]
        g = gpkg[gn]
        kk = list(g.keys())
        ga = np.zeros( ( len(kk), 3 ), dtype=np.object )
        for j in range(len(kk)):
            k = kk[j]  # pkg name
            v = g[k]   # pkg desc
            st = self.stat.desc(k).split(" ")[1]  # code counts
            ga[j] = (k, st, v)   
        pass 
        return gn, ga


    def group_table(self, i):
        n, t = self.group(i)

        wids = [     15,     15,    70  ]  ## lengths of pre and post are auto-added, so do not include them 
        hfmt = [ "%15s", "%15s", "%70s" ]
        rfmt = [ "%15s", "%15s", "%70s" ]
        pre  = ["     " , "     ",  "     " ]
        post = ["     " , "     ",  "     " ]

        labels = self.LABELS
        labels[2] = n 
        tab = RSTTable.Render_(t, self.LABELS, wids, hfmt, rfmt, pre, post )
        tab.colormap = self.ColorMap(); 
        return tab

    def presentation_page(self, groups, title):
        """
        :param groups: list of group indices eg [0,1]
        """
        prefix = " " * 4
        text = ""
        text += title + "\n" 
        text += "-" * len(title)  
        text += "\n\n"

        text += ".. comment\n"
        text += "    created by bin/packages.py:presentation_page with bin/stats.sh \n"
        text += "\n\n"

        for i in groups:
            tab = self.group_table(i)
            s_tab = str(tab)
            text += ".. class:: small" + "\n\n"
            text += textwrap.indent(s_tab, prefix) + "\n\n" 
            text += "\n\n"
        pass
        text += ":g:`green : active development`,   :b:`blue : plan to remove`,   :r:`red : removed`"
        text += "\n\n"
        return text

 

class Stat(object):
    STAT = "/tmp/stats.npy"   # must run bin/stats.sh to create 
    FMT = "hh/cc/cu/py %s/%s/%s/%s"

    def __init__(self):
        self.stat = np.load(self.STAT, allow_pickle=True) 

    def pkgs(self):
        return list(map(str, self.stat[:,0]))    

    def __str__(self):
        return repr(self.stat)     

    def __repr__(self):
        lines = []
        for pkg in self.pkgs():
            lines.append("%s : %s" % (pkg, self.desc(pkg)))
        pass  
        return "\n".join(lines)

    def file_counts(self, pkg):
        """
        array([['npy', 181, 165, 0, 6]], dtype=object)
        """
        w = np.where( self.stat[:,0] == pkg)[0] 
        return self.stat[w][0] if len(w) == 1 else None

    def desc(self, pkg):
        cnt = self.file_counts(pkg)
        return "?" if cnt is None else self.FMT % tuple(cnt[1:])
    
if __name__ == '__main__':

    st = Stat()
    #print(repr(st))

    pkgs = Packages(st)
    print(repr(pkgs))
    #print(str(pkgs))

    p2 = pkgs.presentation_page([2,3], "Opticks Packages : Many Removed, Many Added") 
    print(p2)

    p1 = pkgs.presentation_page([0,1], "Opticks Packages : :b:`Many more to be removed`") 
    print(p1)







