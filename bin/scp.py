#!/usr/bin/env python
"""
Workarounds for working with a remote SVN repo you are not allowed 
to commit into and a remote working copy over a slow connection.

Essentially want to be able to locally "svn up" and make edits  
then scp.py over changed files into the remote working copy 
for compilation and testing.  

Usage::

    ~/opticks/bin/scp.py         ## emit to stdout the scp commands 
    ~/opticks/bin/scp.py | sh    ## pipe them to shell 

"""
import os, commands, re, argparse, logging
log = logging.getLogger(__name__)

class Mod(object):
    pat = re.compile("^(?P<st>\S)\s*(?P<path>\S*)$")

    @classmethod
    def parse_args(cls, doc):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "--rsvnbase", default="P:junotop/offline", help="remote svn working copy" ) 
        parser.add_argument( "--level", default="info", help="logging level" ) 
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        return args

    def __init__(self, args):
        self.args = args 
        self.paths = self.get_paths() 

    def __str__(self):
        rsvnbase = self.args.rsvnbase  
        return "\n".join(map(lambda path:"scp %s %s/%s" % (path,rsvnbase,path),self.paths))

    def get_paths(self):
        """
        Parse the output of "svn status" collecting status strings and paths
        """
        rc, out = commands.getstatusoutput("svn status")
        assert rc == 0 
        paths = []
        for line in out.split("\n"):
            m = self.pat.match(line)
            assert m, line
            d = m.groupdict()
            #print("%(st)s : %(path)s " % (d))
            assert d["st"] in ["M","A"]
            paths.append(d["path"])
        pass
        return paths


if __name__ == '__main__':
    args = Mod.parse_args(__doc__)
    md = Mod(args)
    print(md)






