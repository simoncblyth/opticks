#!/usr/bin/env python
"""
"""
import numpy as np
import os, sys, re, logging, argparse, platform
from opticks.ana.num import slice_, _slice
from opticks.ana.env import opticks_environment
from opticks.ana.OpticksQuery import OpticksQuery 
from opticks.ana.nload import tagdir_
from opticks.ana.log import init_logging

log = logging.getLogger(__name__) 


def isIPython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True

class OK(argparse.Namespace):
    pass
    #ipython = property(lambda self:sys.argv[0].endswith("ipython"))
    ipython = isIPython()
    brief = property(lambda self:"pfx %s tag %s src %s det %s c2max %s ipython %s " % (self.pfx, self.utag,self.src,self.det, self.c2max, self.ipython))

    def _get_ctx(self):
        return dict(tag=self.tag, utag=self.utag, src=self.src, det=self.det)
    ctx = property(_get_ctx)

    def _get_tagdir(self):
        return tagdir_(self.det, self.src, self.tag, pfx=self.pfx )
    tagdir = property(_get_tagdir)

    def _get_ntagdir(self):
        itag = int(self.tag) 
        return tagdir_(self.det, self.src, str(-itag), pfx=self.pfx )
    ntagdir = property(_get_ntagdir)



    def _get_catdir(self):
        return tagdir_(self.det, self.src, None, pfx=self.pfx )
    catdir = property(_get_catdir)

    def _get_username(self):
        """
        Same approach as SSys::username
        """
        k = "USERNAME" if platform.system() == "Windows" else "USER"
        return os.environ[k]
    username = property(_get_username)

    def _get_tmpdefault(self):
        return os.path.join("/tmp", self.username, "opticks")  
    tmpdefault = property(_get_tmpdefault)

    def resolve(self, arg):
        """
        :return: path with $TMP tokens replaced with a TMP envvar OR the default of /tmp/USERNAME/opticks
        """
        token = "$TMP"
        tmpd = os.environ.get(token[1:], self.tmpdefault )
        if arg.find(token) > -1:
            path = arg.replace(token,tmpd)   
        else:
            path = os.path.expandvars(arg)
        pass
        assert path.find("$") == -1, "failed to resolve tokens in arg %s path %s " % (arg, path ) 
        #print("resolve arg %s to path %s " % (arg, path))
        return path


def opticks_args(**kwa):

    oad_key = "OPTICKS_ANA_DEFAULTS"
    oad = os.environ.get(oad_key,"det=g4live,src=natural,tag=1,pfx=.")
    defaults = dict(map(lambda ekv:ekv.split("="), oad.split(","))) 
    lv = os.environ.get("LV", None)

    if lv is not None:
        lv_is_int = re.compile("\d+").match(lv) is not None
        lvn = lv if not lv_is_int else "proxy-%d" % int(lv)
        defaults["pfx"] = "tboolean-%s" % lvn 
        log.info("override pfx default as LV=%s envvar defined, pfx=%s " % (lv, defaults["pfx"])) 
    pass  

    log.info("envvar %s -> defaults %s " % (oad_key, repr(defaults)))

    det = kwa.get("det", defaults["det"])
    src = kwa.get("src", defaults["src"])
    tag = kwa.get("tag", defaults["tag"])
    pfx = kwa.get("pfx", defaults["pfx"])

    llv = kwa.get("loglevel", "info")
    llv2 = kwa.get("log-level", "info")
    mrc = kwa.get("mrc", 101)
    doc = kwa.get("doc", None)
    tagoffset = kwa.get("tagoffset", 0)
    multievent = kwa.get("multievent", 1)
    stag = kwa.get("stag", None)
    ptag = kwa.get("ptag", None)
    show = kwa.get("show", True)
    plot = kwa.get("plot", True)
    terse = kwa.get("terse", False)
    mat = kwa.get("mat", "GdDopedLS")
    msli = kwa.get("msli", "0:100k")   # 0:1M  mmap_mode slice for quick analysis
    sli = kwa.get("sli", "::")
    sel = kwa.get("sel", "0:5:1")
    qwn = kwa.get("qwn", "XYZT,ABCW")

    c2max = kwa.get("c2max", "1.5,2.0,2.5")
    rdvmax = kwa.get("rdvmax", "0.01,0.10,1.0") 
    #pdvmax = kwa.get("pdvmax", "0.0010,0.0200,0.1000") 
    pdvmax = kwa.get("pdvmax", "0.10,0.25,0.50") 
    #dveps = kwa.get("dveps", 0.0002)

    pfxseqhis = kwa.get("pfxseqhis", "")
    pfxseqmat = kwa.get("pfxseqmat", "")
    dbgseqhis = kwa.get("dbgseqhis", "0")
    dbgseqmat = kwa.get("dbgseqmat", "0")
    dbgmskhis = kwa.get("dbgmskhis", "0")
    dbgmskmat = kwa.get("dbgmskmat", "0")
    smry = kwa.get("smry", False)
    dbgzero = kwa.get("dbgzero", False)
    lmx = kwa.get("lmx", 20)
    cmx = kwa.get("cmx", 0)


    prohis = kwa.get("prohis", False)
    promat = kwa.get("promat", False)
    rehist = kwa.get("rehist", False)
    chi2sel = kwa.get("chi2sel", False)
    chi2selcut = kwa.get("chi2selcut", 1.1)
    statcut = kwa.get("statcut", 1000)
    nointerpol = kwa.get("nointerpol", False)
    figsize = kwa.get("figsize", "18,10.2" )
    size = kwa.get("size", "1920,1080,1" )
    position = kwa.get("position", "100,100" )
    yes = kwa.get("yes", False )
    gltfsave = kwa.get("gltfsave", False )
    lvnlist = kwa.get("lvnlist", "" )

    addpath = kwa.get("addpath", "$LOCAL_BASE/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml" )
    apmtddpath = kwa.get("apmtddpath", "$LOCAL_BASE/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-pmt.xml" )
    apmtpathtmpl = kwa.get("apmtpathtmpl", "$OPTICKS_INSTALL_PREFIX/opticksdata/export/DayaBay/GPmt/%(apmtidx)s/GPmt.npy" )
    apmtidx = kwa.get("apmtidx", 2 )

    csgname = kwa.get("csgname", "tboolean-dummy")
    csgpath = kwa.get("csgpath", None)
    #gltfpath = kwa.get("gltfpath", "$TMP/tgltf/tgltf-gdml--.gltf")

    container = kwa.get("container","Rock//perfectAbsorbSurface/Vacuum") 
    testobject = kwa.get("testobject","Vacuum///GlassSchottF2" ) 

    autocontainer = kwa.get("autocontainer","Rock//perfectAbsorbSurface/Vacuum") 
    autoobject = kwa.get("autoobject","Vacuum/perfectSpecularSurface//GlassSchottF2" ) 
    autoemitconfig = kwa.get("autoemitconfig","photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" ) 
    autoseqmap = kwa.get("autoseqmap","TO:0,SR:1,SA:0" )


    gsel = kwa.get("gsel", "/dd/Geometry/PMT/lvPmtHemi0x" ) 
    gidx = kwa.get("gidx", 0 ) 
    gmaxnode = kwa.get("gmaxnode", 0 ) 
    gmaxdepth = kwa.get("gmaxdepth", 0 ) 
    cfordering = kwa.get("cfordering", "self" ) 
    dumpenv = kwa.get("dumpenv", False) 

    parser = argparse.ArgumentParser(doc)

    parser.add_argument(     "--tag",  default=tag, help="tag identifiying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent. Default %(default)s" )
    parser.add_argument(     "--det",  default=det, help="detector geometry: eg g4live, PmtInBox, dayabay. Default %(default)s. "  )
    parser.add_argument(     "--src",  default=src, help="photon source: torch, natural, scintillation OR cerenkov. Default %(default)s " )
    parser.add_argument(     "--pfx",  default=pfx, help="either \"source\" for 1st executable or the name of the executable for subsequent eg \"OKG4Test\". Default %(default)s " )

    parser.add_argument(     "--noshow",  dest="show", default=show, action="store_false", help="switch off dumping commandline "  )
    parser.add_argument(     "--noplot",  dest="plot", default=plot, action="store_false", help="switch off plotting"  )
    parser.add_argument(     "--show",  default=show, action="store_true", help="dump invoking commandline "  )
    parser.add_argument(     "--loglevel", default=llv, help=" set logging level : DEBUG/INFO/WARNING/ERROR/CRITICAL. Default %(default)s." )
    parser.add_argument(     "--log-level", default=llv2, help=" mirror ipython level option to avoid complications with splitting options. Default %(default)s." )

    parser.add_argument(     "--profile",  default=None, help="Unused option allowing argparser to cope with remnant ipython profile option"  )
    parser.add_argument(     "-i", dest="interactive", action="store_true", default=False, help="Unused option allowing argparser to cope with remnant ipython -i option"  )

    parser.add_argument(     "--tagoffset",  default=tagoffset, type=int, help="tagoffset : unsigned offset from tag, identifies event in multivent running. Default %(default)s "  )
    parser.add_argument(     "--multievent",  default=multievent, type=int, help="multievent : unsigned number of events to handle. Default %(default)s "  )
    parser.add_argument(     "--stag",  default=stag, help="S-Polarization tag : identifying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--ptag",  default=ptag, help="P-Polarization tag : identifying a simulation within a specific source and detector geometry, negated tag for Geant4 equivalent" )
    parser.add_argument(     "--mrc",  default=mrc, type=int, help="script return code resulting from missing event files. Default %(default)s "  )
    parser.add_argument(     "--mat",  default=mat, help="material name, used for optical property dumping/plotting. Default %(default)s"  )
    parser.add_argument(     "--sli",  default=sli, help="slice specification delimited by colon. Default %(default)s"  )
    parser.add_argument(     "--msli",  default=msli, help="photon np.load mmap_mode slice specification delimited by colon. Default %(default)s"  )
    parser.add_argument(     "--sel",  default=sel, help="selection slice specification delimited by colon. Default %(default)s"  )
    parser.add_argument(     "--qwn",  default=qwn, help="Quantity by single char, pages delimited by comma eg XYZT,ABCR. Default %(default)s"  )

    parser.add_argument(     "--c2max",  default=c2max, help="Admissable total chi2 deviation in comparisons. Comma delimited triplet of floats for warn/error/fatal levels. Default %(default)s"  )
    parser.add_argument(     "--rdvmax",  default=rdvmax, help="For compressed record data : admissable total absolute deviation in DvTab comparisons. Comma delimited triplet of floats for warn/error/fatal levels. Default %(default)s"  )
    parser.add_argument(     "--pdvmax",  default=pdvmax, help="For uncompressed final photon data : admissable total absolute deviation in DvTab comparisons. Comma delimited triplet of floats for warn/error/fatal levels. Default %(default)s"  )

    parser.add_argument(     "--pfxseqhis",  default=pfxseqhis, help="Seqhis hexstring prefix for spawned selection. Default %(default)s"  )
    parser.add_argument(     "--pfxseqmat",  default=pfxseqmat, help="Seqmat hexstring prefix for spawned selection. Default %(default)s"  )
    parser.add_argument(     "--dbgseqhis",  default=dbgseqhis, help="Seqhis hexstring prefix for dumping. Default %(default)s"  )
    parser.add_argument(     "--dbgseqmat",  default=dbgseqmat, help="Seqmat hexstring prefix for dumping. Default %(default)s"  )
    parser.add_argument(     "--dbgmskhis",  default=dbgmskhis, help="History mask hexstring for selection/dumping. Default %(default)s"  )
    parser.add_argument(     "--dbgmskmat",  default=dbgmskmat, help="Material mask hexstring for selection/dumping. Default %(default)s"  )
    parser.add_argument(     "--figsize",  default=figsize, help="Comma delimited figure width,height in inches. Default %(default)s"  )
    parser.add_argument(     "--size",  default=size, help="Comma delimited figure width,height in inches. Default %(default)s"  )
    parser.add_argument(     "--position",  default=position, help="Comma delimited window position. Default %(default)s"  )
    parser.add_argument(     "--dbgzero",  default=dbgzero, action="store_true", help="Dump sequence lines with zero counts. Default %(default)s"  )
    parser.add_argument(     "--terse", action="store_true", help="less verbose, useful together with --multievent ")
    parser.add_argument(     "--smry", default=smry, action="store_true", help="smry option gives less detailed seqmat and seqhis tables, including the hex strings, useful for dbgseqhis")
    parser.add_argument(     "--pybnd",  action="store_true", help="Avoid error from op binary selection flag. ")
    parser.add_argument(     "--gdml2gltf",  action="store_true", help="Avoid error from op binary selection flag. ")
    parser.add_argument(     "--prohis", default=prohis, action="store_true", help="Present progressively masked seqhis frequency tables for step by step checking. Default %(default)s ")
    parser.add_argument(     "--promat", default=promat, action="store_true", help="Present progressively masked seqmat frequency tables for step by step checking. Default %(default)s ")
    parser.add_argument(     "--rehist", default=rehist, action="store_true", help="Recreate hists rather than loading persisted ones. Default %(default)s ")
    parser.add_argument(     "--chi2sel", default=chi2sel, action="store_true", help="Select histograms by their chi2 sum exceeding a cut, see cfh.py. Default %(default)s ")
    parser.add_argument(     "--chi2selcut", default=chi2selcut, type=float, help="chi2 per degree of freedom cut used to select histograms when using --chi2sel option, see cfh-vi. Default %(default)s ")
    parser.add_argument(     "--statcut", default=statcut, type=int, help="Statistics cut used with --chi2sel option, see cfh-vi Default %(default)s ")
    parser.add_argument(     "--nointerpol", dest="interpol", default=not nointerpol, action="store_false", help="See cfg4/tests/CInterpolationTest.py. Default %(default)s ")
    parser.add_argument(     "--lmx",  default=lmx, type=int, help="Maximum number of lines to present in sequence frequency tables. Default %(default)s "  )
    parser.add_argument(     "--cmx",  default=cmx, type=float, help="When greater than zero used as minimum line chi2 to present in sequence frequency tables. Default %(default)s "  )
    parser.add_argument(     "--apmtpathtmpl", default=apmtpathtmpl, help="Template Path to analytic PMT serialization, see pmt- and ana/pmt/analytic.py. %(default)s ")
    parser.add_argument(     "--apmtidx",      default=apmtidx, type=int, help="PmtPath index used to fill in the template, see pmt- and ana/pmt/analytic.py. %(default)s ")
    parser.add_argument(     "--apmtddpath",   default=apmtddpath, help="Path to detdesc xml file with description of DayaBay PMT, which references other files. %(default)s ")
    parser.add_argument(     "--addpath",   default=addpath, help="Path to detdesc xml file for topdown testing. %(default)s ")
    parser.add_argument(     "--yes", action="store_true", help="Confirm any YES dialogs. %(default)s ")
    parser.add_argument(     "--csgpath",   default=csgpath, help="Directory of the NCSG input serialization. %(default)s ")
    parser.add_argument(     "--csgname",   default=csgname, help="Name of the Directory of the NCSG input serialization. %(default)s ")
    #parser.add_argument(     "--gltfpath",   default=gltfpath, help="Path to glTF json file. %(default)s ")
    parser.add_argument(     "--container",   default=container, help="Boundary specification for container. %(default)s ")
    parser.add_argument(     "--testobject",  default=testobject, help="Boundary specification for testobject. %(default)s ")

    parser.add_argument(     "--autocontainer",   default=autocontainer, help="Boundary specification for test container used with --testauto. %(default)s ")
    parser.add_argument(     "--autoobject",      default=autoobject, help="Boundary specification for test object used with --testauto. %(default)s ")
    parser.add_argument(     "--autoemitconfig",  default=autoemitconfig, help="Emit config from test container used with --testauto. %(default)s ")
    parser.add_argument(     "--autoseqmap",      default=autoseqmap, help="Seqmap for NCSGIntersect testing with --testauto geometry. %(default)s ")

    parser.add_argument(     "--cfordering",  default=cfordering, help="Sort ordering of cf tables, one of max/self/other. %(default)s ")

    parser.add_argument(     "--gsel",  default=gsel, help="GDML node selection, either tree node index integer or LV name prefix, see tboolean-gdml . %(default)s ")
    parser.add_argument(     "--gmaxdepth",  default=gmaxdepth, type=int, help="GDML node depth limit, 0 for no limit, see tboolean-gdml. %(default)s ")
    parser.add_argument(     "--gmaxnode",  default=gmaxnode, type=int, help="GDML node limit including target node, 0 for no limit, see tboolean-gdml. %(default)s ")
    parser.add_argument(     "--gidx",  default=gidx, type=int, help="GDML index to pick target node from within gsel lvn selection, see tboolean-gdml. %(default)s ")
    parser.add_argument(     "--gltfsave", default=gltfsave, action="store_true", help="Save GDML parsed scene as glTF, see analytic/sc.py. %(default)s ")
    parser.add_argument(     "--lvnlist", default=lvnlist, help="Path to file containing list of lv names. %(default)s ")
    parser.add_argument(     "--j1707", action="store_true", help="Bash level option passthru. %(default)s ")
    parser.add_argument(     "--ip", action="store_true", help="Bash level option passthru. %(default)s ")
    parser.add_argument(     "--pdb", action="store_true", help="ipython level option passthru. %(default)s ")
    parser.add_argument(     "--extras", action="store_true", help="Bash level option passthru. %(default)s ")
    parser.add_argument(     "--dumpenv", default=dumpenv, action="store_true", help="Dump enviroment. %(default)s ")
    parser.add_argument(     "--disco", action="store_true", help="Disable container, investigate suspected  inefficient raytrace of objects inside spacious containers. %(default)s ")

    parser.add_argument('nargs', nargs='*', help='nargs : non-option args')


    ok = OK()
    args = parser.parse_args(namespace=ok)
    # dont write to stdout here it messes up tboolean picking ip TESTCONFIG
    init_logging(level=args.loglevel)


    if args.multievent > 1 and args.tagoffset > 0:
         log.fatal("use either --multievent n or --tagoffset o to pick one from multi, USING BOTH --multievent and --tagoffset NOT SUPPORTED  ") 
         sys.exit(1)

    assert args.cfordering in "max self other".split() 


    if args.det != "g4live" and args.pfx != ".":
        args.det = args.pfx 
    pass

    args.c2max = map(float, args.c2max.split(",")) 
    args.rdvmax = map(float, args.rdvmax.split(",")) 
    args.pdvmax = map(float, args.pdvmax.split(",")) 

    if args.multievent > 1:
        args.utags =  map(lambda offset:int(args.tag) + offset, range(args.multievent)) 
        args.utag = args.utags[0]   # backward compat for scripts not supporting multievent yet 
    else:
        try:
           tag = int(args.tag)
        except ValueError:
           tag = map(int,args.tag.split(","))
        pass

        if type(tag) is int:
            args.utag = tag + args.tagoffset 
            args.utags = [args.utag]   
        else:
            args.utag = None
            args.utags = tag  
        pass
    pass

    args.qwns = args.qwn.replace(",","")       
    os.environ.update(OPTICKS_MAIN_QWNS=args.qwns)  # dirty trick to avoid passing ok to objects that need this

    # hexstring -> hexint 
    args.dbgseqhis = int(str(args.dbgseqhis),16) 
    args.dbgseqmat = int(str(args.dbgseqmat),16) 
    args.dbgmskhis = int(str(args.dbgmskhis),16) 
    args.dbgmskmat = int(str(args.dbgmskmat),16) 
    args.figsize = map(float, args.figsize.split(","))


    args.msli = slice_(args.msli)   # from string to slice 
    args.sli = slice_(args.sli)
    args.sel = slice_(args.sel)

    args.apmtpath = args.apmtpathtmpl % dict(apmtidx=str(args.apmtidx))
    log.debug("args.apmtpathtmpl %s args.apmtidx %d -> args.apmtpath %s " % ( args.apmtpathtmpl, args.apmtidx, args.apmtpath ) ) 


    log.debug("args.dbgseqhis [%x] " % args.dbgseqhis) 
    log.debug("args.smry : %s " % args.smry )

    if args.show:
         sys.stderr.write("args: " + " ".join(sys.argv) + "\n")
    pass 
    ok.query = OpticksQuery(os.environ.get("OPTICKS_QUERY",""))
    return ok 


def opticks_main(**kwa):
    ok = opticks_args(**kwa)
    opticks_environment(ok)
    np.set_printoptions(suppress=True, precision=4, linewidth=200)
    return ok


if __name__ == '__main__':
    ok = opticks_main(doc=__doc__)  
    print(ok)
    print(ok.brief)
     

