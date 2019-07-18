#!/usr/bin/env python
"""
bench.py
============

Presents launchAVG times and prelaunch times for groups of Opticks runs
with filtering based on commandline arguments of the runs and the digest 
of the geocache used.

::

    bench.py --include xanalytic --digest f6cc352e44243f8fa536ab483ad390ce
    bench.py --include xanalytic --digest f6
        selecting analytic results for a particular geometry 

    bench.py --include xanalytic --digest 52e --since May22_1030
        selecting analytic results for a particular geometry after some time 

    bench.py --digest 52 --since 6pm

    bench.py --name geocache-bench360
         fullname of the results dir

    bench.py --name 360
         also works with just a tail string, so long as it selects 
         one of the results dirs 


    bench.py --name 360 --runlabel R1
          select runs with runlabel starting R1


::

    ipython -i $(which bench.py) -- --name geocache-bench360 --include xanalytic --include 10240,5760,1


"""
import os, re, logging, sys, argparse
from collections import OrderedDict as odict
import numpy as np
log = logging.getLogger(__name__)

from dateutil.parser import parse
from datetime import datetime
from opticks.ana.datedfolder import DatedFolder, dateparser
from opticks.ana.meta import Meta
from opticks.ana.key import Key


def lol2l(lol):
    l = []
    if lol is not None:
        for ii in lol:
            assert len(ii) == 1, "expecting a list of lists of length 1"
            i = ii[0]
            l.append(i)
        pass
    pass
    return l 


class Bench(object):

    @classmethod
    def Args(cls):
        argline = "bench.py %s" % " ".join(sys.argv[1:])
        print(argline)

        parser = argparse.ArgumentParser(__doc__)

        resultsprefix = "$OPTICKS_RESULTS_PREFIX" if os.environ.has_key("OPTICKS_RESULTS_PREFIX") else "$TMP"  ## equivalent to BOpticksResource::ResolveResultsPrefix
        parser.add_argument( "--resultsdir", default=os.path.join(resultsprefix, "results"), help="Directory path to results" )
        parser.add_argument( "--name", default="bench", help="String at the end of the directory names beneath resultsdir in which to look for results. Default %(default)s.")
        parser.add_argument( "--digest", default=None, help="Select result groups using geocaches with digests that start with the option string")
        parser.add_argument( "--since", default=None, help="Select results from dated folders following the date string provided, eg May22_1030 or 20190522_173746")
        parser.add_argument( "--include", default=None, action='append', nargs='*', help="Select result groups with commandline containing the string provided. ALL the strings when repeated" )
        parser.add_argument( "--exclude", default=None, action='append', nargs='*', help="Select result groupd with commandline NOT containing the string. NOT containing ANY of the strings when repeated" )
        parser.add_argument( "--runlabel", default=None, help="Select result groups with runlabel starting with the string provided." )
        parser.add_argument( "--xrunlabel", default=None, help="Exclude result groups with runlabel starting with the string provided." )
        parser.add_argument( "--metric", default="launchAVG", help="Quantity key to present in comparison tables. Default %(default)s." );
        parser.add_argument( "--other", default="prelaunch000", help="Another quantity key to list in tables. Default %(default)s." );
        parser.add_argument( "--nodirs", dest="dirs", action="store_false", default=True, help="Skip the listing of results dirs, for more compact output." );
        parser.add_argument( "--splay",  action="store_true", default=False, help="Display the example commandline in a more readable but space consuming form." );
        parser.add_argument( "--nosort",  action="store_true", default=False, help="Dont display time sorted." );
        args = parser.parse_args()
        print(args)
        args.argline = argline
        return args  


    def _get_since(self):
        """
        Parse since strings such as 0930, May22_1030 or 20190522_173746 into a datetime  
        """
        args = self.args
        if args.since is not None:
            now = datetime.now()
            default = datetime(now.year, now.month, now.day)
            since = parse(args.since.replace("_"," "), default=default)  
            print("since : %s " % since )
        else:
            since = None
        pass 
    since = property(_get_since) 


    def _get_base(self):
        args = self.args
        if self._base is None:
            rnames = filter( lambda rname:rname.endswith(args.name),  os.listdir(os.path.expandvars(args.resultsdir)) )
            #print(rnames)
            assert len(rnames) == 1, rnames
            rname = rnames[0]

            base = os.path.join( args.resultsdir, rname )
            base = os.path.expandvars(base)
            self._base = base
            print("base %s" % base)
        pass
        return self._base
    base = property(_get_base)


    def get_udirs(self, df):
        args = self.args
        udirs = filter(lambda _:_.endswith(df),self.dirs)
        if args.runlabel is not None:
            udirs = filter(lambda udir:os.path.dirname(udir).startswith(args.runlabel),  udirs)
        pass 
        if args.xrunlabel is not None:
            udirs = filter(lambda udir:not os.path.dirname(udir).startswith(args.xrunlabel),  udirs)
        pass 
        return udirs 

    def find(self, df):
        rgs = filter(lambda rg:rg.df == df,  self.rgs)
        assert len(rgs) == 1
        return rgs[0] 

    def __init__(self, args):
        self.args = args  

        self.metric = args.metric
        self.metric_ = lambda m:float(m.d["OTracerTimes"][args.metric])
        self.other = args.other 
        self.other_ = lambda m:float(m.d["OTracerTimes"][args.other])

        self.argline = args.argline
        self._base = None
        self.findRunGroups()

    def findRunGroups(self):
        """
        Arrange into groups of runs with the same runstamp/datedfolder
        """ 
        dirs, dfolds, dtimes = DatedFolder.find(self.base)
        assert len(dfolds) == len(dtimes) 

        self.dirs = dirs
        self.dfolds = dfolds
        self.dtimes = dtimes

        order = sorted(range(len(dfolds)), key=lambda i:dtimes[i])   ## sorted by run datetimes
        self.order = order

        rgs = []
        for i in order:
            rg = RunGroup.Make(self, i) 
            if rg is None: continue
            rgs.append(rg)
        pass 
        self.rgs = rgs

    def head(self):
        return []

    def body(self):
        return map(repr, self.rgs)

    def tail(self):
        return ["", self.argline]

    def __repr__(self):
        return "\n".join(self.head() + self.body() + self.tail())

    def __len__(self):
        return len(self.rgs)

    def __getitem__(self, i):
        return self.rgs[i] 



labfmt_ = lambda lab:" %30s %10s %10s %10s      %10s " %  lab
rowfmt_ = lambda row:" %30s %10.3f %10.3f %10.3f      %10.3f  %s " % ( row.label, row.metric, row.rfast, row.rslow, row.other, row.absdir )

class RunGroup(object):

    dtype = [ 
          ("index", np.int32),
          ("label", "|S30"),
          ("metric", np.float32),
          ("rfast", np.float32),
          ("rslow", np.float32),
          ("other", np.float32),
          ("absdir", "|S64"),
            ]

    @classmethod
    def Make(cls, b, i):

        df = b.dfolds[i] 
        dt = b.dtimes[i] 
        udirs = b.get_udirs(df) 

        if len(udirs) == 0: return None
        mm,smm,mcmd,geof,key,cmdline = cls.LoadMeta(b, udirs, dt) 
        if key is None: return None

        rg = cls(b, i, df, dt, mm,smm,mcmd,geof,key,cmdline)
        return rg

    @classmethod
    def LoadMeta(cls, b, udirs, dt ):

        mm = [Meta(p, b.base) for p in udirs]
        smm = sorted(mm, key=b.metric_)  
        cmdline = smm[0].d["parameters"]["CMDLINE"]

        selected = cls.Selected( b, dt, cmdline )
        if selected: 
            def key_(m):
                d = m.d["parameters"]
                k0 = d.get("OPTICKS_KEY",None)
                k1 = d.get("KEY",None)
                kk = list(set(filter(None, [k0,k1])))
                assert len(kk) == 1, d
                return kk[0]
            pass 
            groupcommand_ = lambda m:m.d["parameters"].get("GROUPCOMMAND","-")
            geofunc_ = lambda m:m.d["parameters"].get("GEOFUNC","-")

            keys = map(key_, smm)
            mcmds = map(groupcommand_, smm)
            geofs = map(geofunc_, smm)

            assert len(set(mcmds)) == 1, "all OPTICKS_GROUPCOMMAND for a group of runs with same dated folder should be identical" 
            assert len(set(geofs)) == 1, "all OPTICKS_GEOFUNC for a group of runs with same dated folder should be identical" 
            assert len(set(keys)) == 1, "all OPTICKS_KEY for a group of runs with same dated folder should be identical " 

            mcmd = mcmds[0]
            geof = geofs[0]
            key = keys[0]
        else:
            mcmd = None
            geof = None
            key = None
        pass
        return mm,smm,mcmd,geof,key,cmdline 

    @classmethod
    def Selected(cls, b, dt, cmdline):
        args = b.args 
        select = False

        includes = lol2l(args.include)
        excludes = lol2l(args.exclude)

        if len(includes)>0:
            found = list(set(map(lambda include:cmdline.find(include) > -1, includes)))
            if len(found) == 1 and found[0] == True:
                select=True
            else:
                select=False
            pass    
        elif len(excludes)>0:
            notfound = list(set(map(lambda exclude:cmdline.find(exclude) == -1, excludes)))
            if len(notfound) == 1 and notfound[0] == True:
                select=True
            else:
                select=False
            pass    
        elif args.digest is not None and not digest.startswith(args.digest):
            select=False
        elif b.since is not None and not dt > b.since:
            select=False
        else:
            select=True
        pass
        return select 

    def __repr__(self):
        return "\n".join(self.head() + self.body() + self.tail())

    def _get_path(self):
        return os.path.expandvars(os.path.join("$TMP/ana", "%s_%s.png" % ("bench",self.df) )) 
    path = property(_get_path)

    def head(self):
        lines = ["---  GROUPCOMMAND : %s  GEOFUNC : %s " % (self.mcmd, self.geof)] 
        if self.b.args.splay:
            lines.append("\\\n    --".join(self.cmdline.split("--")))
        else:
            lines.append(self.cmdline)
        pass
        k = Key(key=self.key)
        digest = k.digest
        idpath = k.keydir 
        lines.extend([self.name, self.key, idpath, self.labels])
        return lines

    def body(self):
        return map(rowfmt_, self.a )

    def tail(self):
        return map(lambda kv:"  %30s %10.3f " % (kv[0], kv[1]), self.r.items() )

    def make_a(self):
        b = self.b
        args = b.args
        mm = self.mm
        smm = self.smm
        umm = mm if args.nosort else smm
        ffast = b.metric_(smm[0])
        fslow = b.metric_(smm[-1])
        a = np.recarray((len(self.mm),), dtype=self.dtype )
        d = odict() 
        for i, m in enumerate(umm):
            f = b.metric_(m)
            rfast = f/ffast
            rslow = f/fslow
            o = b.other_(m)
            a[i] = (i, m.parentfold, f, rfast, rslow, o, m.absdir )  
            d[m.parentfold] = f 
        pass
        r = odict()
        for k,v in args.ratios.items():
            assert len(v) == 2, v
            num,den = v
            if num in d and den in d:
                r[k] = d[num]/d[den]
            pass
        pass
        return a, d, r

    def __init__(self, b, i, df, dt, mm, smm, mcmd, geof, key, cmdline):
        self.b = b 
        self.args = b.args
        self.base = b.base
        self.metric = b.metric

        self.i = i 
        self.df = df 
        self.dt = dt 
        self.mm = mm
        self.smm = smm
        self.mcmd = mcmd
        self.geof = geof
        self.key = key
        self.cmdline = cmdline
        self.name = "bench%d" % i 

        lab = ( df, b.metric,"rfast", "rslow", b.other)
        self.labels = labfmt_(lab)

        a, d, r = self.make_a()

        self.a = a
        self.d = d
        self.r = r




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    ratios = odict()
    ratios["R0/1_TITAN_V"] = "R0_TITAN_V R1_TITAN_V".split()
    ratios["R0/1_TITAN_RTX"] = "R0_TITAN_RTX R1_TITAN_RTX".split()
    ratios["R1/0_TITAN_V"] = "R1_TITAN_V R0_TITAN_V".split()
    ratios["R1/0_TITAN_RTX"] = "R1_TITAN_RTX R0_TITAN_RTX".split()

    args = Bench.Args()
    args.ratios = ratios

    b = Bench(args)
    print(b)


    
