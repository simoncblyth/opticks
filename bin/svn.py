#!/usr/bin/env python
"""
svn.py / git.py 
=================

git.py is a symbolic link to svn.py that detects its name
to pick the flavor of version control 


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

::

   svn st | perl -ne 'm,\S\s*(\S*), && print "$1\n"' - | xargs md5 % 
   svn st | perl -ne 'm,\S\s*(\S*), && print "$1\n"' - | xargs md5sum % 


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

       ## OR instead do this with : svn.py rup
       ## that can be combined, eg svn.py rup cf 

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
import os, sys, commands, re, argparse, logging, platform
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
    dig = md5()
    with open(path,'rb') as f:  
        for chunk in iter(lambda: f.read(8192),''): 
            dig.update(chunk)
        pass
    pass
    return dig.hexdigest()


def md5sum_alt(path):
    system =  platform.system()
    if system == "Darwin":
        cmd = "md5 -q %s"  ## just outputs the digest 
        rc,out = commands.getstatusoutput(cmd % path)
        assert rc == 0 
        dig = out      
    elif system == "Linux":
        cmd = "md5sum %s"   ## outputs the digest and the path 
        rc,out = commands.getstatusoutput(cmd % path)
        assert rc == 0 
        dig = out.split(" ")[0]
    else:
        dig = None
        assert 0, system
    pass
    return dig 


log = logging.getLogger(__name__)

expand_ = lambda p:os.path.expandvars(os.path.expanduser(p))  

class Path(dict):
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)
        if not 'dig' in self:
            self["dig"] = md5sum(self["path"])  
            if self.get('check', False) == True:
                self["dig_alt"] = md5sum_alt(self["path"])  
                match = self["dig_alt"] == self["dig"]

                fmt = "%(path)s %(dig)s %(dig_alt)s"
                if not match:
                    log.fatal(" check FAIL : " + fmt % self ) 
                else:
                    log.debug(" check OK   : " + fmt % self ) 
                pass
                assert match, (self["dig_alt"], self["dig"])
            pass
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
    lstpat = re.compile("^\s*(?P<st>\S*)\s*(?P<path>\S*)$")

    @classmethod
    def detect_vc_from_dir(cls):
        if os.path.isdir(".svn"):
            vc = "svn"
        elif os.path.isdir(".git"):
            vc = "git"
        else:
            print("FATAL must invoke from svn or git top level working copy directory")
        pass
        #print("detected vc %s " % vc)
        return vc 

    @classmethod
    def detect_vc_from_scriptname(cls):
        scriptname = os.path.basename(sys.argv[0])  
        if scriptname == "svn.py":
            vc = "svn"
        elif scriptname == "git.py":
            vc = "git"
        else:
            assert 0 
        pass
        #print("detected vc %s " % vc)
        return vc

    @classmethod
    def parse_args(cls, doc):
        vc = cls.detect_vc_from_scriptname()
        defaults = {}
        if vc == "svn":
            defaults["chdir"] = "~/junotop/offline" 
            defaults["rbase"] = "P:junotop/offline" 
            defaults["rstatpath"] = "~/rstat.txt" 
            defaults["rstatcmd"] = "ssh P opticks/bin/svn.py"
            defaults["statcmd"] = "svn status"
        elif vc == "git":
            defaults["chdir"] = "~/opticks" 
            defaults["rbase"] = "P:opticks" 
            defaults["rstatpath"] = "~/rstat_opticks.txt" 
            defaults["rstatcmd"] = "ssh P opticks/bin/git.py"
            defaults["statcmd"] = "git status --porcelain"
        else:
            pass
        pass
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "cmd", default=["st"], nargs="*", choices=["rup","loc","rem","st","get","put","cf","cfu","sync", ["st"]], 
            help="command specifying what to do with the working copy" ) 
        parser.add_argument( "--chdir", default=defaults["chdir"], help="chdir here" ) 
        parser.add_argument( "--rstatpath", default=defaults["rstatpath"], help="path to remote status file" ) 
        parser.add_argument( "--rstatcmd", default=defaults["rstatcmd"], help="command to invoke the remote version of this script" )
        parser.add_argument( "--rbase", default=defaults["rbase"], help="remote svn working copy" ) 
        parser.add_argument( "--check", default=False, action="store_true", help="check digest with os alternative md5 or md5sum" ) 
        parser.add_argument( "--ldig", type=int, default=-1, help="length of digest" ) 
        parser.add_argument( "-p", "--priority", choices=["loc","rem"], default="loc", help="Which version wins when a file exists at both ends" ) 
        parser.add_argument( "--level", default="info", help="logging level" ) 
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        args.chdir = expand_(args.chdir)
        args.rstatpath = expand_(args.rstatpath)

        args.vc = vc
        args.statcmd = defaults["statcmd"]
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
            if line.startswith("Warning: Permanently added"): continue
            if len(line) == 3: continue  # skip the rem/loc title
            m = cls.rstpat.match(line)
            assert m, line
            d = m.groupdict()
            d["ldig"] = ldig
            paths.append(Path(d))
        pass
        return cls(paths, "rem")

    @classmethod         
    def FromStatus(cls, args):
        """
        :param args:
        :return loc: WC instance

        Parse the output of "svn status" collecting status strings and paths
        """

        log.debug("ldig %s check %s statcmd %s " % (args.ldig,args.check, args.statcmd))

        rc, out = commands.getstatusoutput(args.statcmd)
        assert rc == 0 
      
        log.debug(out)  

        paths = []
        for line in filter(None,out.split("\n")):
            log.debug("[%s]"%line)
            m = cls.lstpat.match(line)
            assert m, line
            d = m.groupdict()
            d["ldig"] = args.ldig
            d["check"] = args.check 
            assert d["st"] in ["M","A", "?", "??"]

            if d["st"] == "??": d["st"] = "?"   # bring git into line

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
    def PutCmd(cls, path, rbase, chdir):
        return "scp %s/%s %s/%s" % (chdir,path,rbase,path)

    @classmethod
    def GetCmd(cls, path, rbase, chdir):
        return "scp %s/%s %s/%s" % (rbase,path,chdir,path)

    def scp_put_cmds(self, rbase, chdir):
        """put from local to remote"""
        return "\n".join(map(lambda d:self.PutCmd(d["path"],rbase,chdir), self.paths))

    def scp_get_cmds(self, rbase, chdir):
        """get from remote to local"""
        return "\n".join(map(lambda d:self.GetCmd(d["path"],rbase,chdir), self.paths))

    def _get_hdr(self):
        name = getattr(self, 'name', "noname")
        return "%s" % name

    hdr = property(_get_hdr)

    def __str__(self):
        return "\n".join([self.hdr]+map(str,self.paths))


if __name__ == '__main__':
    args = WC.parse_args(__doc__)

    if "rup" in args.cmd or "cfu" in args.cmd:
        log.info("running args.rstatcmd : %s " % args.rstatcmd )
        rc,out = commands.getstatusoutput(args.rstatcmd)
        assert rc == 0, rc
        #print(out)
        log.info("writing out to args.rstatpath : %s " % args.rstatpath)
        file(args.rstatpath,"w").write(out)
    pass

    if os.path.exists(args.rstatpath):
        rem = WC.FromRemoteStatusFile(args.rstatpath, args.ldig)
    else:
        rem = None
    pass 

    os.chdir(args.chdir)
    loc = WC.FromStatus(args)

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
            print(loc.scp_put_cmds(args.rbase, args.chdir))
        elif cmd == "get": # scp remote to local 
            print(rem.scp_get_cmds(args.rbase, args.chdir))
        elif cmd == "cf" or cmd == "cfu":
            assert cf
            print(cf)
        elif cmd == "sync":
            assert cf
            for p in cf.paths:
                print(str(p))
                stlr = p["stlr"]
                stdig = p["stdig"]
                if stlr == "l ":
                    print(WC.PutCmd(p["path"], args.rbase, args.chdir))
                elif stlr == " r":
                    print(WC.GetCmd(p["path"], args.rbase, args.chdir))
                elif stlr == "lr":
                    if stdig == "*": 
                        if args.priority == "rem":
                            print(WC.GetCmd(p["path"], args.rbase, args.chdir))
                        elif args.priority == "loc":
                            print(WC.PutCmd(p["path"], args.rbase, args.chdir))
                        else:
                            assert 0, args.priority
                        pass
                    pass
                else:
                    assert 0, stlr
                pass
            pass
        elif cmd == "rup":
            pass
        else:
            assert 0, cmd
        pass
    pass

