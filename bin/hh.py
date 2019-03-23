#!/usr/bin/env python
"""
hh.py : Extracting RST documentation embedded into header files
==================================================================

::

    [blyth@localhost opticks]$ cat g4ok/G4Opticks.hh | hh.py --stdin   ## extract docstring from header

    [blyth@localhost ~]$ hh.py  ## examine all headers, looking for ones with docstrings missing 




"""
import re, os, sys, logging, argparse
log = logging.getLogger(__name__)


class Proj(object):

    API_EXPORT_HH = "_API_EXPORT.hh"

    @classmethod
    def rootdir(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log.info("root %s " % root)    
        return root 

    @classmethod
    def find_projs(cls):
        """
        Find project directories relative to root that contain 
        files with names like: OKGEO_API_EXPORT.hh
        """ 
        root = cls.rootdir() 
        projs=[]
        for dirpath, dirs, names in os.walk(root):
            apiexports = filter(lambda name:name.endswith(cls.API_EXPORT_HH), names) 
            reldir = dirpath[len(root)+1:]
            if len(apiexports) > 0:
                assert len(apiexports) == 1
                name = apiexports[0][0:-len(cls.API_EXPORT_HH)]           
                p = cls(reldir, name, root) 
                print(repr(p))
                projs.append(p) 
            pass
        pass
        return projs 

    def __init__(self, reldir, name, root):
        self.reldir = reldir
        self.absdir = os.path.join(root, reldir)
        self.name = name   
        self.hhd = self.find_hhd() 

    def find_hhd(self):
        names = filter(lambda hdrname:hdrname.endswith(".hh") or hdrname.endswith(".hpp"), os.listdir(self.absdir)) 
        names = filter(lambda hdrname:not hdrname.startswith(self.name+"_"), names)   

        hhd = {}
        for name in names:
            path = os.path.join(self.absdir, name)
            lines = open(path, "r").readlines()   
            hh = HH(lines)
            if len(hh.content) == 0:
                print("no docstring in %s " % path)
            pass
            hhd[name] = hh  
        pass 
        return hhd

    def __repr__(self):
        return " %30s : %15s : %d " % (self.reldir, self.name, len(self.hhd))
 



class HH(object):

    BEG = "/**"
    END = "**/"

    def __init__(self, lines):
        self.lines = lines
        self.content = self.extract_content(lines) 

    def extract_content(self, lines):
        """
        Collects content from region 2, to exclude the begin line
        """
        content = []
        region = 0   
        for l in lines:
            c = self.classify(l) 
            if c == "B":
                region = 1 
            elif c == "E":  
                region = 0
            else:
                pass
            pass  
            if region == 2:
                content.append(l)           
            pass 
            if region == 1:
                region += 1   
            pass
            pass
        pass
        return content

    def classify(self, line):
        """
        Note only top level (tight to left edge) comment markers qualify
        """
        if line.startswith(self.BEG): 
            return "B"
        elif line.startswith(self.END):
            return "E"
        else:
            return " "
        pass      

    def __str__(self):
        return "\n".join(["HH",""]+self.lines)
    def __repr__(self):
        return "\n".join(self.content)







if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stdin",  action="store_true", help="Read header file on stdin and extract the docstring" )
    args = parser.parse_args()
 
    if args.stdin:
        lines = map(str.rstrip, sys.stdin.readlines())
        hh = HH(lines)
        print(repr(hh))
    else:
        Proj.find_projs()
    pass





