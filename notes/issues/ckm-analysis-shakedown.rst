ckm-analysis-shakedown
=========================


Ana scripts currently unaware of geocache evt saving
------------------------------------------------------

* base.py still using IDPATH : update for OPTICKS_KEY 

::

    [blyth@localhost ana]$ unset IDPATH
    [blyth@localhost ana]$ ./base.py 
    ana/base.py:OpticksEnv missing IDPATH envvar [$IDPATH] 
    [blyth@localhost ana]$



::

     10 import os, sys, logging, numpy as np
     11 log = logging.getLogger(__name__)
     12 
     13 from opticks.ana.base import opticks_main
     14 from opticks.ana.evt import Evt
     15 
     16 if __name__ == '__main__':
     17     args = opticks_main(tag="1",src="natural", det="g4live", doc=__doc__)
     18     np.set_printoptions(suppress=True, precision=3)
     19 
     20     evt = Evt(tag=args.tag, src=args.src, det=args.det, seqs=[], args=args)
     21 
     22     log.debug("evt")
     23     print evt
     24 
     26     evt.history_table(slice(0,20))




    [blyth@localhost opticks]$ ckm.py
    args: /home/blyth/opticks/ana/ckm.py
    [2019-05-28 09:22:39,591] p227351 {/home/blyth/opticks/ana/base.py:598} WARNING - failed to load json from /tmp/blyth/opticks/evt/g4live/natural/1/parameters.json
    [2019-05-28 09:22:39,591] p227351 {/home/blyth/opticks/ana/base.py:582} CRITICAL - failed to load ini from /tmp/blyth/opticks/evt/g4live/natural/1/t_delta.ini
    Traceback (most recent call last):
      File "/home/blyth/opticks/ana/ckm.py", line 20, in <module>
        evt = Evt(tag=args.tag, src=args.src, det=args.det, seqs=[], args=args)
      File "/home/blyth/opticks/ana/evt.py", line 201, in __init__
        ok = self.init_metadata(tag, src, det, dbg)
      File "/home/blyth/opticks/ana/evt.py", line 288, in init_metadata
        fdom = A.load_("fdom",src,tag,det, dbg=dbg) 
      File "/home/blyth/opticks/ana/nload.py", line 200, in load_
        raise IOError("cannot load %s " % path)
    IOError: cannot load /tmp/blyth/opticks/evt/g4live/natural/1/fdom.npy 
    [blyth@localhost opticks]$ 




GFlags/abbrev.json coming from OPTICKS_DATA_DIR ?  FIXED
----------------------------------------------------------------

* lifecycle of the abbrev.json should follow that of the enum flags, and 
  be persisted beside each other 

  * DONE : moved into installcache/OKC


hismask.py::

 19 class HisMask(MaskType):
 20     """ 
 21     Why are resource/GFlags/abbrev.json coming from OPTICKS_DATA_DIR ?
 22     installcache seems more appropriate ?
 23     """ 
 24     def __init__(self):
 25         log.debug("HisMask.__init__")
 26         flags = EnumFlags()
 27         abbrev = Abbrev("$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json")
 28         MaskType.__init__(self, flags, abbrev)
 29         log.debug("HisMask.__init__ DONE")
 30 
 31 

::

    [blyth@localhost ana]$ grep OPTICKS_DATA_DIR *.py 
    base.py:        * extracate use of OPTICKS_DATA_DIR, used by hismask.py for flag abbreviations
    base.py:        #self.setdefault("OPTICKS_DATA_DIR",        os.path.join(self.install_prefix, "opticksdata"))   
    base.py:            self.setdefault("OPTICKS_DATA_DIR",        _dirname(IDPATH,3))
    base.py:            self.setdefault("OPTICKS_DATA_DIR",        _dirname(self.srcpath,3))     ## HUH thats a top down dir, why go from bottom up for it ?
    base.py:    $OPTICKS_DATA_DIR/resource/GFlags/abbrev.json
    dbgseed.py:        g = np.load(x_("$OPTICKS_DATA_DIR/gensteps/%s/%s/%s.npy" % (args.det,args.src,args.tag) ))
    hismask.py:    Why are resource/GFlags/abbrev.json coming from OPTICKS_DATA_DIR ?
    hismask.py:        abbrev = Abbrev("$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json")
    histype.py:        abbrev = Abbrev("$OPTICKS_DATA_DIR/resource/GFlags/abbrev.json")
    [blyth@localhost ana]$ 
    [blyth@localhost ana]$ 


::

    [blyth@localhost ana]$ cd ~/local/opticks/installcache/
    [blyth@localhost installcache]$ l
    total 8
    drwxrwxr-x. 3 blyth blyth 4096 May 27 21:01 PTX
    drwxrwxr-x. 2 blyth blyth  108 Jul  6  2018 OKC
    drwxrwxr-x. 2 blyth blyth   78 Jul  6  2018 RNG
    [blyth@localhost installcache]$ cd OKC
    [blyth@localhost OKC]$ l
    total 16
    -rw-rw-r--. 1 blyth blyth 274 Jul  6  2018 GFlagIndexLocal.ini
    -rw-rw-r--. 1 blyth blyth 274 Jul  6  2018 GFlagIndexSource.ini
    -rw-rw-r--. 1 blyth blyth 274 Jul  6  2018 GFlagsLocal.ini
    -rw-rw-r--. 1 blyth blyth 274 Jul  6  2018 GFlagsSource.ini
    [blyth@localhost OKC]$ cat GFlagsSource.ini
    BOUNDARY_REFLECT=11
    BOUNDARY_TRANSMIT=12
    BULK_ABSORB=4
    BULK_REEMIT=5
    BULK_SCATTER=6
    CERENKOV=1
    EMITSOURCE=19
    FABRICATED=16
    G4GUN=15
    MACHINERY=18
    MISS=3
    NAN_ABORT=14
    NATURAL=17
    SCINTILLATION=2
    SURFACE_ABSORB=8
    SURFACE_DETECT=7
    SURFACE_DREFLECT=9
    SURFACE_SREFLECT=10
    TORCH=13
    [blyth@localhost OKC]$ pwd
    /home/blyth/local/opticks/installcache/OKC
    [blyth@localhost OKC]$ cat ../../opticksdata/resource/GFlags/abbrev.json
    {
        "CERENKOV":"CK",
        "SCINTILLATION":"SI",
        "TORCH":"TO",
        "MISS":"MI",
        "BULK_ABSORB":"AB",
        "BULK_REEMIT":"RE", 
        "BULK_SCATTER":"SC",    
        "SURFACE_DETECT":"SD",
        "SURFACE_ABSORB":"SA",      
        "SURFACE_DREFLECT":"DR",
        "SURFACE_SREFLECT":"SR",
        "BOUNDARY_REFLECT":"BR",
        "BOUNDARY_TRANSMIT":"BT",
        "NAN_ABORT":"NA"
    }


optickscore/OpticksFlags.cc::

     16 //const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/graphics/optixrap/cu/photon.h" ;
     17 //const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/opticks/OpticksPhoton.h" ;
     18 //const char* OpticksFlags::ENUM_HEADER_PATH = "$ENV_HOME/optickscore/OpticksPhoton.h" ;
     19 
     20 const char* OpticksFlags::ENUM_HEADER_PATH = "$OPTICKS_INSTALL_PREFIX/include/OpticksCore/OpticksPhoton.h" ;
     21 //  envvar OPTICKS_INSTALL_PREFIX is set internally by OpticksResource based on cmake config 
     22 
     23 
     24 const char* OpticksFlags::ZERO_              = "." ;
     25 const char* OpticksFlags::NATURAL_           = "NATURAL" ;
     26 const char* OpticksFlags::FABRICATED_        = "FABRICATED" ;
     27 const char* OpticksFlags::MACHINERY_         = "MACHINERY" ;
     28 const char* OpticksFlags::EMITSOURCE_        = "EMITSOURCE" ;
     29 const char* OpticksFlags::PRIMARYSOURCE_     = "PRIMARYSOURCE" ;
     30 const char* OpticksFlags::GENSTEPSOURCE_     = "GENSTEPSOURCE" ;
     31 
     32 const char* OpticksFlags::CERENKOV_          = "CERENKOV" ;
     33 const char* OpticksFlags::SCINTILLATION_     = "SCINTILLATION" ;
     34 const char* OpticksFlags::MISS_              = "MISS" ;
     35 const char* OpticksFlags::OTHER_             = "OTHER" ;
     36 const char* OpticksFlags::BULK_ABSORB_       = "BULK_ABSORB" ;
     37 const char* OpticksFlags::BULK_REEMIT_       = "BULK_REEMIT" ;
     38 const char* OpticksFlags::BULK_SCATTER_      = "BULK_SCATTER" ;
     39 const char* OpticksFlags::SURFACE_DETECT_    = "SURFACE_DETECT" ;
     40 const char* OpticksFlags::SURFACE_ABSORB_    = "SURFACE_ABSORB" ;
     41 const char* OpticksFlags::SURFACE_DREFLECT_  = "SURFACE_DREFLECT" ;
     42 const char* OpticksFlags::SURFACE_SREFLECT_  = "SURFACE_SREFLECT" ;
     43 const char* OpticksFlags::BOUNDARY_REFLECT_  = "BOUNDARY_REFLECT" ;
     44 const char* OpticksFlags::BOUNDARY_TRANSMIT_ = "BOUNDARY_TRANSMIT" ;
     45 const char* OpticksFlags::TORCH_             = "TORCH" ;
     46 const char* OpticksFlags::G4GUN_             = "G4GUN" ;
     47 const char* OpticksFlags::NAN_ABORT_         = "NAN_ABORT" ;
     48 const char* OpticksFlags::BAD_FLAG_          = "BAD_FLAG" ;
     49 
     50 // NB this is duplicating abbrev from /usr/local/opticks/opticksdata/resource/GFlags/abbrev.json
     51 //    TODO: get rid of that 
     52 //
     53 //     as these are so fixed they deserve static enshrinement for easy access from everywhere
     54 //
     55 const char* OpticksFlags::_ZERO              = "  " ;
     56 const char* OpticksFlags::_NATURAL           = "NL" ;
     57 const char* OpticksFlags::_FABRICATED        = "FD" ;
     58 const char* OpticksFlags::_MACHINERY         = "MY" ;
     59 const char* OpticksFlags::_EMITSOURCE        = "SO" ;
     60 const char* OpticksFlags::_PRIMARYSOURCE     = "PS" ;
     61 const char* OpticksFlags::_GENSTEPSOURCE     = "GS" ;
     62 
     63 const char* OpticksFlags::_CERENKOV          = "CK" ;
     64 const char* OpticksFlags::_SCINTILLATION     = "SI" ;
     65 const char* OpticksFlags::_TORCH             = "TO" ;
     66 const char* OpticksFlags::_MISS              = "MI" ;
     67 const char* OpticksFlags::_BULK_ABSORB       = "AB" ;
     68 const char* OpticksFlags::_BULK_REEMIT       = "RE" ;
     69 const char* OpticksFlags::_BULK_SCATTER      = "SC" ;
     70 const char* OpticksFlags::_SURFACE_DETECT    = "SD" ;
     71 const char* OpticksFlags::_SURFACE_ABSORB    = "SA" ;
     72 const char* OpticksFlags::_SURFACE_DREFLECT  = "DR" ;
     73 const char* OpticksFlags::_SURFACE_SREFLECT  = "SR" ;
     74 const char* OpticksFlags::_BOUNDARY_REFLECT  = "BR" ;
     75 const char* OpticksFlags::_BOUNDARY_TRANSMIT = "BT" ;
     76 const char* OpticksFlags::_NAN_ABORT         = "NA" ;
     77 const char* OpticksFlags::_G4GUN             = "GN" ;
     78 const char* OpticksFlags::_BAD_FLAG          = "XX" ;
     79 



Added OpticksFlags::getAbbrevMeta, now where to persist the json ?
-----------------------------------------------------------------------

::

     880 opticks-prepare-installcache()
     881 {
     882     local msg="=== $FUNCNAME :"
     883     echo $msg generating RNG seeds into installcache 
     884 
     885     cudarap-
     886     cudarap-prepare-installcache
     887 
     888     OpticksPrepareInstallCache_OKC
     889 }


* tests/OpticksPrepareInstallCache_OKC.cc

::

     04 int main(int argc, char** argv )
      5 {
      6     OPTICKS_LOG(argc, argv);
      7     
      8     Opticks ok(argc, argv) ;
      9     ok.configure();
     10     ok.Summary();
     11     
     12     if(argc > 1 && strlen(argv[1]) > 0)
     13     {
     14         LOG(warning) << "WRITING TO MANUAL PATH IS JUST FOR TESTING" ;
     15         ok.prepareInstallCache(argv[1]);
     16     }   
     17     else
     18     {
     19         ok.prepareInstallCache();
     20     }   
     21     
     22     return 0 ;
     23 }   

::

    2647 /**
    2648 Opticks::prepareInstallCache
    2649 -----------------------------
    2650 
    2651 Moved save directory from IdPath to ResourceDir as
    2652 the IdPath is not really appropriate  
    2653 for things such as the flags that are a feature of an 
    2654 Opticks installation, not a feature of the geometry.
    2655 
    2656 But ResourceDir is not appropriate either as that requires 
    2657 manual management via opticksdata repo.
    2658 
    2659 
    2660 **/ 
    2661 
    2662 
    2663 void Opticks::prepareInstallCache(const char* dir)
    2664 {
    2665     if(dir == NULL) dir = m_resource->getOKCInstallCacheDir() ;
    2666     LOG(info) << ( dir ? dir : "NULL" )  ; 
    2667     m_resource->saveFlags(dir);
    2668     m_resource->saveTypes(dir);
    2669 }   


* save the OpticksFlagsAbbrevMeta.json together with flags



mattype.py reading from "$OPTICKS_DETECTOR_DIR/GMaterialLib/abbrev.json"
----------------------------------------------------------------------------

* added GEOCACHE metadata to OpticksEvent, can assert on this matching IDPATH
* thence can read the abbrev from the GEOCACHE


::

    /home/blyth/opticks/ana/mattype.pyc in __init__(self, reldir)
        103     def __init__(self, reldir=None):
        104         material_names = ItemList("GMaterialLib", reldir=reldir)
    --> 105         material_abbrev = Abbrev("$OPTICKS_DETECTOR_DIR/GMaterialLib/abbrev.json")
        106         SeqType.__init__(self, material_names, material_abbrev)
        107 



detector detector_dir OPTICKS_DETECTOR_DIR make no sense in direct workflow
-------------------------------------------------------------------------------------------

The main useful thing I recall was the ordering of materials
so that relevant materials occupy the low numbers, allowing history recording 
with 4 bits per material.

BUT : can just say they need to change the Geant4 ordering to change the Opticks
ordering 


base.py::

    140 class OpticksEnv(object):
    ...
    191     def _detector(self):
    192         """
    193         does not make any sense in direct approach
    194         """
    195         idname = self._idname()
    196         dbeg = idname.split("_")[0]
    197         if dbeg in ["DayaBay","LingAo","Far"]:
    198             detector =  "DayaBay"
    199         else:
    200             detector = dbeg
    201         pass
    202         log.debug("_opticks_detector idpath %s -> detector %s " % (self.idpath, detector))
    203         return detector
    204 
    205     def _detector_dir(self):
    206         """
    207         in layout 1, this yields /usr/local/opticks/opticksdata/export/juno1707/
    208         but should be looking in IDPATH ?
    209         """
    210         detector = self._detector()
    211         return os.path.join(self.env["OPTICKS_EXPORT_DIR"], detector)
    212 



::

    [blyth@localhost export]$ cd DayaBay
    [blyth@localhost DayaBay]$ l
    total 4
    drwxrwxr-x. 2 blyth blyth  42 Jul  5  2018 GSurfaceLib
    drwxrwxr-x. 3 blyth blyth  15 Jul  5  2018 GPmt
    drwxrwxr-x. 2 blyth blyth  61 Jul  5  2018 GMaterialLib
    -rw-rw-r--. 1 blyth blyth 862 Jul  5  2018 ChromaMaterialMap.json
    [blyth@localhost DayaBay]$ 
    [blyth@localhost DayaBay]$ 
    [blyth@localhost DayaBay]$ l GSurfaceLib/
    total 8
    -rw-rw-r--. 1 blyth blyth  317 Jul  5  2018 color.json
    -rw-rw-r--. 1 blyth blyth 1466 Jul  5  2018 order.json
    [blyth@localhost DayaBay]$ l GPmt/
    total 0
    drwxrwxr-x. 2 blyth blyth 223 Jul  5  2018 0
    [blyth@localhost DayaBay]$ l GPmt/0/
    total 40
    -rw-rw-r--. 1 blyth blyth   74 Jul  5  2018 GPmt_lvnames.txt
    -rw-rw-r--. 1 blyth blyth   47 Jul  5  2018 GPmt_materials.txt
    -rw-rw-r--. 1 blyth blyth   74 Jul  5  2018 GPmt_pvnames.txt
    -rw-rw-r--. 1 blyth blyth  289 Jul  5  2018 GPmt_check.txt
    -rw-rw-r--. 1 blyth blyth 1168 Jul  5  2018 GPmt_csg.npy
    -rw-rw-r--. 1 blyth blyth   47 Jul  5  2018 GPmt_csg.txt
    -rw-rw-r--. 1 blyth blyth  289 Jul  5  2018 GPmt_boundaries.txt
    -rw-rw-r--. 1 blyth blyth  848 Jul  5  2018 GPmt_check.npy
    -rw-rw-r--. 1 blyth blyth  848 Jul  5  2018 GPmt.npy
    -rw-rw-r--. 1 blyth blyth  289 Jul  5  2018 GPmt.txt
    [blyth@localhost DayaBay]$ l GMaterialLib/
    total 12
    -rw-rw-r--. 1 blyth blyth 660 Jul  5  2018 color.json
    -rw-rw-r--. 1 blyth blyth 795 Jul  5  2018 order.json
    -rw-rw-r--. 1 blyth blyth 612 Jul  5  2018 abbrev.json
    [blyth@localhost DayaBay]$ pwd
    /home/blyth/local/opticks/opticksdata/export/DayaBay
    [blyth@localhost DayaBay]$ 
    [blyth@localhost DayaBay]$ 
    [blyth@localhost DayaBay]$ cd ..



Two character Material Abbreviation in direct workflow ?
----------------------------------------------------------

* automated abbrev 

  * when mix of upper and lower case use 1st 2 uppers  
  * 1st two uppercased chars
  * with fallback to 1st and last, or 1st 2 

* perhaps with option to use own abbreviations given to Opticks with  

  * G4Opticks::setMaterialAbbreviations(path_to_abbrev.json)


::

    [blyth@localhost DayaBay]$ cat GMaterialLib/abbrev.json 
    {
        "ADTableStainlessSteel": "AS",
        "Acrylic": "Ac",
        "Air": "Ai",
        "Aluminium": "Al",
        "Bialkali": "Bk",
        "DeadWater": "Dw",
        "ESR": "ES",
        "Foam": "Fo",
        "GdDopedLS": "Gd",
        "IwsWater": "Iw",
        "LiquidScintillator": "LS",
        "MineralOil": "MO",
        "Nitrogen": "Ni",
        "NitrogenGas": "NG",
        "Nylon": "Ny",
        "OwsWater": "Ow",
        "PPE": "PP",
        "PVC": "PV",
        "Pyrex": "Py",
        "Rock": "Rk",
        "StainlessSteel": "SS",
        "Tyvek": "Ty",
        "UnstStainlessSteel": "US",
        "Vacuum": "Vm",
        "OpaqueVacuum": "OV",
        "Water": "Wt",
        "GlassSchottF2": "F2"
    }



Implemented automated abbrev with two sysrap classes::

    SASCII
    SAbbrev

Now that its automated, should be applied in X4 together 
with other material translation as part of the population
of GGeo.

Just needs material names, so better to do internally 
inside GMaterialLib.


::

    234 void X4PhysicalVolume::convertMaterials()
    235 {
    236     OK_PROFILE("_X4PhysicalVolume::convertMaterials");
    237     LOG(verbose) << "[" ;
    238 
    239     size_t num_materials0 = m_mlib->getNumMaterials() ;
    240     assert( num_materials0 == 0 );
    241 
    242     X4MaterialTable::Convert(m_mlib);
    243 
    244     size_t num_materials = m_mlib->getNumMaterials() ;
    245     assert( num_materials > 0 );
    246 
    247     // Adding test materials only at Opticks level is a standardization
    248     // problem : TODO: implement creation of test materials at G4 level
    249     // then they will be present at all levels.
    250     // 
    251     //m_mlib->addTestMaterials() ;
    252 
    253     m_mlib->close();   // may change order if prefs dictate
    254 
    255     LOG(verbose) << "]" ;
    256     LOG(info)
    257           << " num_materials " << num_materials
    258           ;
    259     OK_PROFILE("X4PhysicalVolume::convertMaterials");
    260 }



::

     456 NMeta* GMaterialLib::createMeta()
     457 {
     458     LOG(LEVEL) << "." ;
     459     NMeta* libmeta = new NMeta ;
     460     unsigned int ni = getNumMaterials();
     461 
     462     std::vector<std::string> names ;
     463     for(unsigned int i=0 ; i < ni ; i++)
     464     {
     465         GMaterial* mat = m_materials[i] ;
     466         const char* name = mat->getShortName();
     467         names.push_back(name);
     468     }
     469 
     470     SAbbrev abbrev(names);
     471     assert( abbrev.abbrev.size() == names.size() );
     472     assert( abbrev.abbrev.size() == ni );
     473 
     474     NMeta* abbrevmeta = new NMeta ;
     475     for(unsigned i=0 ; i < ni ; i++)
     476     {
     477         const std::string& nm = names[i] ;
     478         const std::string& ab = abbrev.abbrev[i] ;
     479         abbrevmeta->set<std::string>(nm.c_str(), ab ) ;
     480     }
     481 
     482     libmeta->setObj("abbrev", abbrevmeta );
     483     return libmeta ;
     484 }


::

    [blyth@localhost 1]$ l GMaterialLib/
    total 8
    -rw-rw-r--. 1 blyth blyth 3824 May 28 18:57 GMaterialLib.npy
    -rw-rw-r--. 1 blyth blyth   49 May 28 18:57 GPropertyLibMetadata.json
    [blyth@localhost 1]$ jsn.py GMaterialLib/GPropertyLibMetadata.json
    {u'abbrev': {u'Air': u'Ai', u'Glass': u'Gl', u'Water': u'Wa'}}
    [blyth@localhost 1]$ 




