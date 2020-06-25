#!/usr/bin/env python
"""
svn.py
========

This script enables two svn working copies "local" and "remote"
to be kept in sync with each other without requiring the changes
to be committed to svn.  It provides a workaround for operating with 
a remote svn working copy over a slow connection when you do not yet
have permission to commit many of the changes.

This slim script avoids the need to wield the "git svn" sledgehammer
to crack a nut.

Essentially want to be able to locally "svn up" and make edits  
then selectively scp over changed files (with different digests) 
into the remote working copy for compilation and testing.  

Generally it is best to avoid editing on the remote end, but it is sometimes 
unavoidable. This script eases the pain of bringing both working copies
back in sync without having to commit the changes.

NB all the below commands do no harm, they only suggest the scp commands 
that need to be manually run in the shell or piped there.

Workflow::

   loc> export PATH=$HOME/opticks/bin:$PATH

   loc> svn up    
   rem> svn up

   loc> vi ...   
       ## local editing, adding files 

   loc> scp ~/opticks/bin/svn.py P:opticks/bin/svn.py 
       ## update this script at remote 

   loc> ssh P opticks/bin/svn.py > ~/rstat.txt     
       ## take snapshot of remote working copy digests 

   loc> svn.py loc   
       ## list status of local working copy with file digests  
   loc> svn.py rem   
       ## ditto for remote, using the ~/rstat.txt snapshot from above
   loc> svn.py cf    
       ## compare status showing which files are only loc/rem and which have different digests

   loc> svn.py put
       ## emit to stdout scp commands to local to remote copy wc files of M/A/? status    
   loc> svn.py get
       ## emit to stdout scp commands to remote to local copy wc files of M/A/? status    
   loc> svn.py sync
       ## show the cf output interleaved with put/get commands to bring the two wc together
       ##
       ## NB where there are digest changes, no command is suggested as it is necessary to 
       ## manually examine the differences to see which is ahead OR to merge changes 
       ## from both ends if there has been a mixup and changes were made in the wrong file 


   loc> svn.py sync -p rem | grep scp 
       ## with remote priority, show the sync scp commands 

   loc> svn.py sync -p rem | grep scp | sh 
       ## pipe those commands to shell

"""
import os, commands, re, argparse, logging
from collections import OrderedDict as odict
try: 
    from hashlib import md5 
except ImportError: 
    from md5 import md5 
pass

def md5sum_py3(path):
    with open(path, mode='rb') as f:
        d = md5()
        for buf in iter(partial(f.read, 4096), b''):
            d.update(buf)
        pass
    return d.hexdigest()

def md5sum(path):
    f = open(path, mode='rb')
    d = md5()
    for buf in f.read(4096):
        d.update(buf)
    return d.hexdigest()

log = logging.getLogger(__name__)

expand_ = lambda p:os.path.expandvars(os.path.expanduser(p))  

class Path(dict):
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)
        if not 'dig' in self:
            self["dig"] = md5sum(self["path"])        
        pass
        if not 'dig5' in self:
            self["dig5"] = self["dig"][:5]
        pass
    def __str__(self):
        dig = self["dig"]
        ldig = self.get("ldig", -1)
        if ldig < 0: ldig = len(dig)
        return "%1s %s %s " % (self["st"], dig[:ldig], self["path"])

class WC(object):
    rstpat = re.compile("^(?P<st>\S)\s*(?P<dig>\S*)\s*(?P<path>\S*)$")
    lstpat = re.compile("^(?P<st>\S)\s*(?P<path>\S*)$")

    @classmethod
    def parse_args(cls, doc):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "cmd", default=["st"], nargs="*", choices=["loc","rem","st","get","put","cf","sync", ["st"]], 
            help="command specifying what to do with the working copy" ) 
        parser.add_argument( "--chdir", default="~/junotop/offline", help="chdir here" ) 
        parser.add_argument( "--rstatpath", default="~/rstat.txt", help="path to remote status file" ) 
        parser.add_argument( "--rsvnbase", default="P:junotop/offline", help="remote svn working copy" ) 
        parser.add_argument( "--ldig", type=int, default=-1, help="length of digest" ) 
        parser.add_argument( "-p", "--priority", choices=["loc","rem"], default="rem", help="Which version wins when a file exists at both ends" ) 
        parser.add_argument( "--level", default="info", help="logging level" ) 
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        args.chdir = expand_(args.chdir)
        args.rstatpath = expand_(args.rstatpath)
        return args

    @classmethod
    def FromRemoteStatusFile(cls, rstatpath, ldig):
        """
        :param rstatpath:
        :param ldig: int length of digest
        :return rem: WC instance

        Parse the status output of a remote instance of this script
        """
        log.debug("reading %s " % rstatpath) 
        lines = map(str.rstrip, file(rstatpath, "r").readlines())
        paths = []
        for line in lines:
            if len(line) == 3: continue  # skip the rem/loc title
            m = cls.rstpat.match(line)
            assert m, line
            d = m.groupdict()
            d["ldig"] = ldig
            paths.append(Path(d))
        pass
        return cls(paths, "rem")

    @classmethod         
    def FromStatus(cls, ldig):
        """
        :param ldig: int length of digest
        :return loc: WC instance

        Parse the output of "svn status" collecting status strings and paths
        """
        rc, out = commands.getstatusoutput("svn status")
        assert rc == 0 
        paths = []
        for line in out.split("\n"):
            m = cls.lstpat.match(line)
            assert m, line
            d = m.groupdict()
            d["ldig"] = ldig
            assert d["st"] in ["M","A", "?"]
            if os.path.isdir(d["path"]):
                log.debug("skip dir %s " % d["path"] )
            else:    
                paths.append(Path(d))
            pass
        pass
        return cls(paths, "loc")

    @classmethod         
    def FromComparison(cls, loc, rem, ldig):
        """
        :param loc: WC instance
        :param rem: WC instance
        :param ldig: length of digest
        :return cf: WC instance
        """
        l = loc.d
        r = rem.d 
        u = set(l).union(set(r))
        paths = []
        stfmt = "%2s %1s%1s %1s"
        dgfmt = "%5s|%5s" 

        index_ = lambda ls,val:ls.index(val) if val in ls else -1 
       
        for k in sorted(list(u), key=lambda k:max(index_(l.keys(),k),index_(r.keys(),k))):
            st = "".join(["l" if k in l else " ","r" if k in r else " "])
            rk = r.get(k, None)
            lk = l.get(k, None)

            d = dict(path=k)
            d["ldig"] = ldig

            stdig = " "

            if st == "lr":
                stdig = "=" if lk["dig"] == rk["dig"] else "*"
                stdat = (st, lk["st"], rk["st"], stdig )
                d["dig"] = "%s|%s" % (lk["dig"],rk["dig"])
                d["dig5"] = dgfmt % (lk["dig5"],rk["dig5"] )
            elif st == "l ":
                stdat = (st, lk["st"], "", " " )
                d["dig"] = "%s|%s" % (lk["dig"],"-" * 32 )
                d["dig5"] = dgfmt % (lk["dig5"], "-" * 5 )
            elif st == " r":
                stdat = (st, "", rk["st"], " " )
                d["dig"] = "%s|%s" % ("-" * 32, rk["dig"])
                d["dig5"] = dgfmt % ("-" * 5, rk["dig5"] )
            pass
            d["st"] = stfmt % stdat  
            d["stlr"] = st 
            d["stdig"] = stdig
            paths.append(Path(d))
        pass
        return cls(paths, "cf")

    def __init__(self, paths, name):
        self.paths = paths
        self.name = name
        d = odict()
        for p in paths:
            d[p["path"]] = p
        pass
        self.d = d

    @classmethod
    def PutCmd(cls, path, rsvnbase, chdir):
        return "scp %s/%s %s/%s" % (chdir,path,rsvnbase,path)

    @classmethod
    def GetCmd(cls, path, rsvnbase, chdir):
        return "scp %s/%s %s/%s" % (rsvnbase,path,chdir,path)

    def scp_put_cmds(self, rsvnbase, chdir):
        """put from local to remote"""
        return "\n".join(map(lambda d:self.PutCmd(d["path"],rsvnbase,chdir), self.paths))

    def scp_get_cmds(self, rsvnbase, chdir):
        """get from remote to local"""
        return "\n".join(map(lambda d:self.GetCmd(d["path"],rsvnbase,chdir), self.paths))

    def _get_hdr(self):
        name = getattr(self, 'name', "noname")
        return "%s" % name

    hdr = property(_get_hdr)

    def __str__(self):
        return "\n".join([self.hdr]+map(str,self.paths))


if __name__ == '__main__':
    args = WC.parse_args(__doc__)

    if os.path.exists(args.rstatpath):
        rem = WC.FromRemoteStatusFile(args.rstatpath, args.ldig)
    else:
        rem = None
    pass 

    os.chdir(args.chdir)
    loc = WC.FromStatus(args.ldig)

    if loc and rem:
        cf = WC.FromComparison(loc,rem, args.ldig)
        #cf = None
    else:
        cf = None
    pass

    for cmd in args.cmd:
        log.debug(cmd)
        if cmd == "loc" or cmd == "st":
            print(loc)
        elif cmd == "rem":
            print(rem)
        elif cmd == "put": # scp local to remote
            print(loc.scp_put_cmds(args.rsvnbase, args.chdir))
        elif cmd == "get": # scp remote to local 
            print(rem.scp_get_cmds(args.rsvnbase, args.chdir))
        elif cmd == "cf":
            assert cf
            print(cf)
        elif cmd == "sync":
            assert cf
            for p in cf.paths:
                print(str(p))
                stlr = p["stlr"]
                stdig = p["stdig"]
                if stlr == "l ":
                    print(WC.PutCmd(p["path"], args.rsvnbase, args.chdir))
                elif stlr == " r":
                    print(WC.GetCmd(p["path"], args.rsvnbase, args.chdir))
                elif stlr == "lr":
                    if stdig == "*": 
                        if args.priority == "rem":
                            print(WC.GetCmd(p["path"], args.rsvnbase, args.chdir))
                        elif args.priority == "loc":
                            print(WC.PutCmd(p["path"], args.rsvnbase, args.chdir))
                        else:
                            assert 0, args.priority
                        pass
                    pass
                else:
                    assert 0, stlr
                pass
            pass
        else:
            assert 0, cmd
        pass
    pass

