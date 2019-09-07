#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""
hh.py : Extracting RST documentation embedded into header files
==================================================================

::

    [blyth@localhost opticks]$ cat g4ok/G4Opticks.hh | hh.py --stdin   ## extract docstring from header

    [blyth@localhost ~]$ hh.py  ## examine all headers, looking for ones with docstrings missing 




"""
import re, os, sys, logging, argparse
log = logging.getLogger(__name__)

def prepdir(odir):
    if not os.path.exists(odir):
        os.makedirs(odir)   
    pass



class Index(list):

    TAIL = r"""

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

"""


    PFX = " " * 3
    def __init__(self, *args, **kwa ):
        title = kwa.pop("title", "no-title") 
        maxdepth = kwa.pop("maxdepth", None) 
        caption = kwa.pop("caption", None) 
        top = kwa.pop("top", False) 
        list.__init__(self, *args, **kwa)
        self.title = title 
        self.maxdepth = maxdepth 
        self.caption = caption
        self.top = top

    def __str__(self):
        tl = lambda _:self.PFX+_
        head = [self.title, "=" * len(self.title), "", ".. toctree::"]

        if not(self.maxdepth is None):
            head.append(tl(":maxdepth: %s" % self.maxdepth))
        pass
        if not(self.caption is None):
            head.append(tl(":caption: %s" % self.caption))
        pass
        body = map(tl, self ) 
        space = [""] 
        s = "\n".join( head + space + body + space )

        if self.top:
           s += self.TAIL 
        pass
        return s 


    def save(self, path="index.rst"):
        with open(path,"w") as fp:
            fp.write(str(self))
        pass


    



class Root(object):

    @classmethod
    def rootdir(cls):
        """
        :return: Opticks root directory, based on known depth of this script
        """  
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log.info("root %s " % root)    
        return root 

    @classmethod
    def find_projs(cls, root):
        """
        Find project directories relative to root that contain 
        files with names like: OKGEO_API_EXPORT.hh

        Create Proj instances for these directories.
        """ 
        projs=[]
        for dirpath, dirs, names in os.walk(root):
            apiexports = filter(lambda name:name.endswith(Proj.API_EXPORT_HH), names) 
            reldir = dirpath[len(root)+1:]
            if len(apiexports) > 0:
                assert len(apiexports) == 1
                name = apiexports[0][0:-len(Proj.API_EXPORT_HH)]           
                p = Proj(reldir, name, root) 
                print(repr(p))
                projs.append(p) 
            pass
        pass
        return projs 


    def __init__(self):
        pass
        self.root = self.rootdir()
        self.projs = self.find_projs(self.root)

    def __repr__(self):
        return "Root %s %d " % ( self.root, len(self.projs)) 

    def __str__(self):
        return "\n".join( [repr(self), "" ] + map(repr, self.projs) )

    def save(self, obase=None):
        if obase is None:
            obase = os.path.expandvars("/tmp/$USER/opticks/hh")
        pass
        log.info("obase %s " % obase)

        title = os.path.basename(self.root)
   
        projs = filter( lambda p:len(p.hhd) > 0, self.projs )
        stems = map(lambda p:"%s/index" % p.reldir, projs)
        idx = Index(stems, title=title, maxdepth=1, caption="Contents:", top=True)

        prepdir(obase)
        os.chdir(obase)
        idx.save(path="index.rst")

        print(str(idx)) 

        for p in projs:
            odir = os.path.join(obase, p.reldir)
            prepdir(odir)
            os.chdir(odir)
            p.save(odir) 
        pass


class Proj(object):

    API_EXPORT_HH = "_API_EXPORT.hh"

    def __init__(self, reldir, name, root):
        """
        :param reldir: relative to Opticks root directory 
        :param name: of project directory 
        :param root: directory 
        """
        self.reldir = reldir
        self.absdir = os.path.join(root, reldir)
        self.name = name   
        self.hhd = self.find_hhd() 

    def find_cc(self, hhpath):
        assert os.path.exists(hhpath), hhpath
        if hhpath.endswith(".hh"):
            ccpath = hhpath[:-3]+".cc"
        elif hhpath.endswith(".hpp"):
            ccpath = hhpath[:-4]+".cpp"
        else:
            pass  
        pass
        exists = os.path.exists(ccpath)
        return ccpath if exists else None

    def find_hhd(self):
        """
        :return hhd: dict keyed on header name holding header information held within HH instances
        """
        names = filter(lambda hdrname:hdrname.endswith(".hh") or hdrname.endswith(".hpp"), os.listdir(self.absdir)) 
        names = filter(lambda hdrname:not hdrname.startswith(self.name+"_"), names)   

        hhd = {}
        for name in names:
            hdr = os.path.join(self.absdir, name)
            imp = self.find_cc(hdr)

            lines = []
            lines += open(hdr, "r").readlines()   

            if imp is not None:
                lines += open(imp, "r").readlines(); 
            pass

            hh = HH(lines)
            if len(hh.content) == 0:
                log.debug("no docstring in hdr %s imp %s  " % (hdr, imp) )
            else:
                hhd[name] = hh  
            pass
        pass 
        return hhd

    def __repr__(self):
        return " %30s : %15s : %d " % (self.reldir, self.name, len(self.hhd))

    def save(self, odir):
        log.info("\nProj %s " % odir)

        names = self.hhd.keys()
        stems = map(lambda name:os.path.splitext(name)[0], names)
        title = "%s : %s " % (self.name, self.reldir)
        idx = Index(stems, title=title)
        print(str(idx)) 
        idx.save(path="index.rst")  

        for k in names:
            stem = os.path.splitext(k)[0]
            hh = self.hhd[k]
            hh.save("%s.rst" % stem)
        pass


class HH(object):

    BEG = "/**"
    END = "**/"

    def __init__(self, lines):
        """
        :param lines: full header text
        """ 
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

    def save(self, path):
        with open(path, "w") as fp:
            fp.write("".join(self.content))
        pass



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
        root = Root()
        print(repr(root))
        root.save()
    pass





