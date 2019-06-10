opticks-event-paths
====================



analysis needs adjusting for proxy locations
------------------------------------------------

::

    2019-06-10 11:45:13.436 INFO  [154811] [OpticksAna::run@89] OpticksAna::run anakey tboolean enabled Y
    args: /home/blyth/opticks/ana/tboolean.py --tag 1 --tagoffset 0 --det tboolean-proxy-17 --src torch
    [2019-06-10 11:45:13,931] p155002 {/home/blyth/opticks/ana/tboolean.py:63} INFO - pfx tboolean-box tag 1 src torch det tboolean-proxy-17 c2max 2.0 ipython False 
    [2019-06-10 11:45:13,932] p155002 {/home/blyth/opticks/ana/ab.py:109} INFO - ab START
    [2019-06-10 11:45:13,932] p155002 {/home/blyth/opticks/ana/nload.py:105} ERROR - NON EXISTING tagdir : /tmp/tboolean-box/evt/tboolean-proxy-17/torch/1  expanded from $OPTICKS_EVENT_BASE/$0/evt/$1/$2/$3 
    [2019-06-10 11:45:13,932] p155002 {/home/blyth/opticks/ana/nload.py:106} ERROR - As relative paths are used with test geometry running, subsequent scripts or executables that intend to reuse data should be invoked from the same directory.
    Traceback (most recent call last):
      File "/home/blyth/opticks/ana/tboolean.py", line 65, in <module>
        ab = AB(ok)
      File "/home/blyth/opticks/ana/ab.py", line 115, in __init__
        self.load()
      File "/home/blyth/opticks/ana/ab.py", line 137, in load
        a = Evt(tag=atag, src=args.src, det=args.det, pfx=args.pfx, args=args, nom="A", smry=args.smry)
      File "/home/blyth/opticks/ana/evt.py", line 187, in __init__
        self.tagdir = tagdir_(det, src, tag, pfx=pfx)
      File "/home/blyth/opticks/ana/nload.py", line 107, in tagdir_
        assert 0, (xdir, tmpl, DEFAULT_DIR_TEMPLATE)
    AssertionError: ('/tmp/tboolean-box/evt/tboolean-proxy-17/torch/1', '$OPTICKS_EVENT_BASE/tboolean-box/evt/tboolean-proxy-17/torch/1', '$OPTICKS_EVENT_BASE/$0/evt/$1/$2/$3')
    2019-06-10 11:45:13.971 INFO  [154811] [SSys::run@70] tboolean.py --tag 1 --tagoffset 0 --det tboolean-proxy-17 --src torch   rc_raw : 256 rc : 1
    2019-06-10 11:45:13.971 WARN  [154811] [SSys::run@77] SSys::run FAILED with  cmd tboolean.py --tag 1 --tagoffset 0 --det tboolean-proxy-17 --src torch  


Hmm python and C++ event addressing has diverged a bit.
BOpticksResource is using evtbase and srcevtbase to do what python uses the pfx for eg "source"::

    621     m_srcevtbase = makeIdPathPath("source");
    622     m_res->addDir( "srcevtbase", m_srcevtbase );
    623 
    624     const char* exename = SProc::ExecutableName() ;
    625     bool exename_allowed = SStr::EndsWith(exename, "Test") || SStr::EndsWith(exename, "Minimal") ;
    626     if(!exename_allowed)
    627     {
    628         LOG(fatal) << "exename " << exename
    629                    << " is not allowed "
    630                    << " (this is to prevent stomping on geocache content). "
    631                    << " Names ending with Test or Minimal are permitted"
    632                    ;
    633     }
    634     assert( exename_allowed );
    635 
    636     //const char* user = SSys::username(); 
    637     //m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;  
    638 
    639     m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath(exename ) ;
    640     ///  should this always be KeySource ???
    641     ///      NO : KeySource means that the current executable is same as the exename 
    642     ///           enshrined into the geocache : ie the geocache creator  
    643 

* so in that lingo the exename is the pfx unless its "source"

::

    158 GMergedMesh* GGeoTest::initCreateCSG()
    159 {
    160     assert(m_csgpath && "m_csgpath is required");
    161     assert(strlen(m_csgpath) > 3 && "unreasonable csgpath strlen");
    162 
    163     m_resource->setTestCSGPath(m_csgpath); // take note of path, for inclusion in event metadata
    164     m_resource->setTestConfig(m_config_); // take note of config, for inclusion in event metadata
    165     m_resource->setEventBase(m_csgpath);   // BResource("evtbase") yields OPTICKS_EVENT_BASE 
    166 

::

    2019-06-10 13:51:58.555 INFO  [364740] [GGeoTest::init@129] [
    2019-06-10 13:51:58.555 FATAL [364740] [GGeoTest::initCreateCSG@163]  m_csgpath tboolean-proxy-17
    2019-06-10 13:51:58.556 INFO  [364740] [BResource::set@99] evtbase change /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/OKG4Test to /tmp/tboolean-proxy-17




changes : moved to 5 argument form
------------------------------------------------------

::

    2019-06-10 17:18:29.845 INFO  [328112] [OGeo::convertMergedMesh@235] ( 0
    2019-06-10 17:18:29.846 INFO  [328112] [OGeo::makeOGeometry@543] ugeocode [A]
    2019-06-10 17:18:29.846 INFO  [328112] [OGeo::makeAnalyticGeometry@617] mm 0 verbosity: 0   pts:  GParts  primflag         flagnodetree numParts    2 numPrim    2
    2019-06-10 17:18:29.872 INFO  [328112] [OGeo::convertMergedMesh@267] ) 0 numInstances 1
    2019-06-10 17:18:29.887 INFO  [328112] [OGeo::convert@230] ] nmm 1
    2019-06-10 17:18:29.888 INFO  [328112] [OScene::init@157] ]
    2019-06-10 17:18:30.768 INFO  [328112] [SLog::operator@28]  ) OKPropagator::OKPropagator  DONE
    2019-06-10 17:18:30.768 INFO  [328112] [SLog::operator@28]  ) OKG4Mgr::OKG4Mgr  DONE
    2019-06-10 17:18:30.770 INFO  [328112] [OpticksRun::setGensteps@148] genstep 1,6,4
    2019-06-10 17:18:30.770 ERROR [328112] [G4StepNPY::checklabel@243]  i 0 unexpected label 262144
    2019-06-10 17:18:30.770 FATAL [328112] [G4StepNPY::checklabel@276] G4StepNPY::checklabel FAIL xlabel 1 ylabel 2 numStep 1 mismatch 1
    OKG4Test: /home/blyth/opticks/npy/G4StepNPY.cpp:284: void G4StepNPY::checklabel(int, int): Assertion `mismatch == 0' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe2023207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2023207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20248f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe201c026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe201c0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffe513cca0 in G4StepNPY::checklabel (this=0x8027980, xlabel=1, ylabel=2) at /home/blyth/opticks/npy/G4StepNPY.cpp:284
    #5  0x00007fffe5635eac in OpticksRun::importGenstepData (this=0x6b6b70, gs=0x6424bb0, oac_label=0x0) at /home/blyth/opticks/optickscore/OpticksRun.cc:338
    #6  0x00007fffe563502e in OpticksRun::setGensteps (this=0x6b6b70, gensteps=0x6424bb0) at /home/blyth/opticks/optickscore/OpticksRun.cc:156
    #7  0x00007ffff7bd56e3 in OKG4Mgr::propagate_ (this=0x7fffffffcca0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:172
    #8  0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcca0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #9  0x00000000004039a7 in main (argc=31, argv=0x7fffffffcfd8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 


Hmm, why suddenly takes exception to EMITSOURCE gensteps ?::

    In [7]: 0x1 << 18
    Out[7]: 262144

    [blyth@localhost optickscore]$ cat OpticksPhoton.h 
    #pragma once

    enum
    {
        CERENKOV          = 0x1 <<  0,    
        SCINTILLATION     = 0x1 <<  1,    
        ...
        EMITSOURCE        = 0x1 << 18,
        PRIMARYSOURCE     = 0x1 << 19,
        GENSTEPSOURCE     = 0x1 << 20
    }; 




Suspect being confused by what it thinks are direct gensteps::

    2019-06-10 19:04:21.549 INFO  [63778] [OpticksHub::loadGeometry@534] ]
    2019-06-10 19:04:21.550 INFO  [63778] [nnode::selectSheets@1606] nnode::selectSheets nsa 6 sheetmask 1 ns 1
    2019-06-10 19:04:21.550 INFO  [63778] [BRng::setSeed@45] U setSeed(42)
    2019-06-10 19:04:21.550 INFO  [63778] [BRng::setSeed@45] V setSeed(43)
    2019-06-10 19:04:21.561 INFO  [63778] [OpticksGen::initFromDirectGensteps@159] /tmp/blyth/opticks/evt/tboolean-box/torch/1/gs.npy
    2019-06-10 19:04:21.561 FATAL [63778] [SLog::SLog@12]  ( CG4::CG4 

Prevent that by isTest check::

     32 OpticksGen::OpticksGen(OpticksHub* hub)
     33     :
     34     m_hub(hub),
     35     m_gun(new OpticksGun(hub)),
     36     m_ok(hub->getOpticks()),
     37     m_cfg(m_ok->getCfg()),
     38     m_ggb(hub->getGGeoBase()),
     39     m_blib(m_ggb->getBndLib()),
     40     m_lookup(hub->getLookup()),
     41     m_torchstep(NULL),
     42     m_fabstep(NULL),
     43     m_csg_emit(hub->findEmitter()),
     44     m_dbgemit(m_ok->isDbgEmit()),  
     45     m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, EMITSOURCE, m_ok->getSeed(), m_dbgemit, m_ok->getMaskBuffer(), m_ok->getGenerateOverride() ) : NULL ),
     46     m_input_photons(NULL),
     47     m_direct_gensteps(m_ok->hasKey() && m_ok->existsDirectGenstepPath() && !m_ok->isTest() ? m_ok->loadDirectGenstep() : NULL ),
     48     m_legacy_gensteps(NULL),
     49     m_source_code(initSourceCode())
     50 {
     51     init() ;
     52 }











::

    164 void OKG4Mgr::propagate_()
    165 {
    166     bool align = m_ok->isAlign();
    167 
    168 
    169     if(m_generator->hasGensteps())   // TORCH
    170     {
    171          NPY<float>* gs = m_generator->getGensteps() ;
    172          m_run->setGensteps(gs);
    173 
    174          if(align)
    175              m_propagator->propagate();
    176 
    177          m_g4->propagate();
    178     }
    179     else   // no-gensteps : G4GUN or PRIMARYSOURCE
    180     {
    181          NPY<float>* gs = m_g4->propagate() ;
    182 
    183          if(!gs) LOG(fatal) << "CG4::propagate failed to return gensteps" ;
    184          assert(gs);
    185 
    186          m_run->setGensteps(gs);
    187     }
    188 
    189     if(!align)
    190         m_propagator->propagate();
    191 }






review
-----------


4-argument form of NPY save::

    1591 void OpticksEvent::saveDomains()
    1592 {
    1593     updateDomainsBuffer();
    1594 
    1595     NPY<float>* fdom = getFDomain();
    1596     if(fdom) fdom->save(fdom_, m_typ,  m_tag, m_udet);
    1597 
    1598     NPY<int>* idom = getIDomain();
    1599     if(idom) idom->save(idom_, m_typ,  m_tag, m_udet);
    1600 }

* 1st argument is the stem of the .npy name eg "fdom", "idom", "so", "ox", "ph", "no", "rx"

::

     730 template <typename T>
     731 void NPY<T>::save(const char* tfmt, const char* source, const char* tag, const char* det)
     732 {
     733     //std::string path_ = NPYBase::path(det, source, tag, tfmt );
     734     std::string path_ = BOpticksEvent::path(det, source, tag, tfmt );
     735     save(path_.c_str());
     736 }


See BOpticksEventTest::

    [BOpticksEvent::directory_@80]  top tboolean-box sub torch tag 1 anno NULL base (directory_template) $OPTICKS_EVENT_BASE/evt/$1/$2/$3


The root depends on OPTICKS_EVENT_BASE::

    [blyth@localhost tests]$ opticks-f OPTICKS_EVENT_BASE
    ./ana/nload.py:#DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"
    ./ana/nload.py:DEFAULT_BASE = "$OPTICKS_EVENT_BASE/$0/evt"
    ./ana/ncensus.py:    c = Census("$OPTICKS_EVENT_BASE/source/evt")
    ./ana/base.py:        #self.setdefault("OPTICKS_EVENT_BASE",      os.path.join(keydir, "source" ))
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",       keydir )
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
    ./boostrap/tests/BFileTest.cc:    ss.push_back("$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1") ; 
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:       LOG(debug) << "BOpticksEvent::directory_template OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
    ./boostrap/BOpticksEvent.cc:       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    ./boostrap/BFile.cc:    else if(strcmp(key,"OPTICKS_EVENT_BASE")==0) 
    ./boostrap/BFile.cc:        LOG(verbose) << "replacing $OPTICKS_EVENT_BASE  with " << evalue ; 
    [blyth@localhost opticks]$ 



OPTICKS_EVENT_BASE is not an allowed envvar, so it defaults to $TMP unless evtbase was set::

    159 std::string BFile::ResolveKey( const char* key )
    160 {
    161 
    162     const char* envvar = SSys::getenvvar(key) ;
    163     std::string evalue ;
    164 
    165     if( IsAllowedEnvvar(key) )
    166     {
    167         if( envvar != NULL )
    168         {
    169             evalue = envvar ;
    170             LOG(verbose) << "replacing allowed envvar token " << key << " with value of tenvvar " << evalue ;
    171         }
    172         else
    173         {
    174             evalue = usertmpdir("/tmp","opticks", NULL);
    175             LOG(error) << "replacing allowed envvar token " << key << " with default value " << evalue << " as envvar not defined " ;
    176         }
    177     }
    178     else if(strcmp(key,"KEYDIR")==0 )
    179     {
    180         const char* idpath = BResource::Get("idpath") ;
    181         assert( idpath );
    182         evalue = idpath ;
    183         LOG(error) << "replacing $IDPATH with " << evalue ;
    184     }
    185     else if(strcmp(key,"DATADIR")==0 )
    186     {
    187         const char* datadir = BResource::Get("opticksdata_dir") ;
    188         assert( datadir );
    189         evalue = datadir ;
    190         LOG(error) << "replacing $DATADIR with " << evalue ;
    191     }
    192     else if(strcmp(key,"OPTICKS_EVENT_BASE")==0)
    193     {
    194         const char* evtbase = BResource::Get("evtbase") ;
    195         if( evtbase != NULL )
    196         {
    197             evalue = evtbase ;
    198         }
    199         else
    200         {
    201             //evalue = BResource::Get("tmpuser_dir") ; 
    202             evalue = usertmpdir("/tmp","opticks",NULL);
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
    203         }
    204         LOG(verbose) << "replacing $OPTICKS_EVENT_BASE  with " << evalue ;
    205     }
    206     else
    207     {
    208         evalue = key ;
    209     } 
    210     return evalue ;
    211 }



evtbase::

    [blyth@localhost opticks]$ opticks-f evtbase
    ./boostrap/BOpticksEvent.cc:srcevtbase
    ./boostrap/BOpticksEvent.cc:    const char* srcevtbase = BResource::Get("srcevtbase");   
    ./boostrap/BOpticksEvent.cc:    if( srcevtbase == NULL ) srcevtbase = BResource::Get("tmpuser_dir") ;   
    ./boostrap/BOpticksEvent.cc:    assert( srcevtbase ); 
    ./boostrap/BOpticksEvent.cc:    std::string path = BFile::FormPath(srcevtbase, "evt", det, typ, tag ); 
    ./boostrap/BOpticksResource.cc:    m_srcevtbase(NULL),
    ./boostrap/BOpticksResource.cc:    m_evtbase(NULL),
    ./boostrap/BOpticksResource.cc:    m_srcevtbase = makeIdPathPath("source"); 
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "srcevtbase", m_srcevtbase ); 
    ./boostrap/BOpticksResource.cc:    //m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;  
    ./boostrap/BOpticksResource.cc:    m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath(exename ) ;  
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "evtbase", m_evtbase ); 
    ./boostrap/BOpticksResource.cc:  it writes its event and genstep into a distinctive "standard" directory (resource "srcevtbase") 
    ./boostrap/BOpticksResource.cc:  a relpath named after the executable (resource "evtbase")   
    ./boostrap/BOpticksResource.cc:srcevtbase 
    ./boostrap/BOpticksResource.cc:evtbase
    ./boostrap/BOpticksResource.cc://const char* BOpticksResource::getSrcEventBase() const { return m_srcevtbase ; } 
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getEventBase() const { return m_evtbase ; } 
    ./boostrap/BOpticksResource.hh:        const char* m_srcevtbase ; 
    ./boostrap/BOpticksResource.hh:        const char* m_evtbase ; 
    ./boostrap/BFile.cc:        const char* evtbase = BResource::Get("evtbase") ; 
    ./boostrap/BFile.cc:        if( evtbase != NULL )
    ./boostrap/BFile.cc:            evalue = evtbase ; 
    [blyth@localhost opticks]$ 




Missing::

    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib/GPropertyLibMetadata.json


    [blyth@localhost 1]$ cat GMaterialLib/GPropertyLibMetadata.json
    {"abbrev":{"Air":"Ai","Glass":"Gl","Water":"Wa"}}

Created it with GMaterialLibTest:create_Meta


Added capability to change evtbase, used this for test geometry  
-----------------------------------------------------------------

::

    162 GMergedMesh* GGeoTest::initCreateCSG()
    163 {
    164     assert(m_csgpath && "misconfigured");
    165     assert(strlen(m_csgpath) > 3 && "unreasonable csgpath strlen");
    166 
    167     m_resource->setTestCSGPath(m_csgpath); // take note of path, for inclusion in event metadata
    168     m_resource->setTestConfig(m_config_); // take note of config, for inclusion in event metadata
    169     m_resource->setEventBase(m_csgpath);   // BResource("evtbase") yields OPTICKS_EVENT_BASE 
    170 



Profile saving is using OpticksEventSpec::getEventFold which doesnt honour evtbase changes : FIXED
-------------------------------------------------------------------------------------------------------

Fixed by moving Opticks::defineEventSpec from configure which was too early 
into a new Opticks::postgeometry 



::

    1735     m_profile->setDir(getEventFold());  // from Opticks::configure (from m_spec (OpticksEventSpec)

    [blyth@localhost optickscore]$ OpticksEventSpecTest
    2019-06-02 21:16:24.784 INFO  [362461] [OpticksEventSpec::Summary@148] s0 (no cat) typ typ tag tag itag 0 det det cat (null) dir /tmp/blyth/opticks/evt/det/typ/tag
    2019-06-02 21:16:24.784 INFO  [362461] [OpticksEventSpec::Summary@148] s1 (with cat) typ typ tag tag itag 0 det det cat cat dir /tmp/blyth/opticks/evt/cat/typ/tag


::

     60 void OpticksEventSpec::init()
     61 {
     62     const char* udet = getUDet();
     63     std::string tagdir = NLoad::directory(udet, m_typ, m_tag ) ;
     64     std::string reldir = NLoad::reldir(udet, m_typ, m_tag ) ;
     65     std::string typdir = NLoad::directory(udet, m_typ, NULL ) ;
     66     m_dir = strdup(tagdir.c_str());
     67     m_reldir = strdup(reldir.c_str());
     68     m_fold = strdup(typdir.c_str());
     69 }
     70 


::

    NLoadTest

    2019-06-02 21:46:56.564 INFO  [410962] [test_directory@31]  NLoad::directory("det", "typ", "tag", "anno" ) /tmp/blyth/opticks/evt/det/typ/tag/anno
    2019-06-02 21:46:56.564 INFO  [410962] [test_reldir@37]  NLoad::reldir("det", "typ", "tag" ) evt/det/typ/tag
    [blyth@localhost tests]$ 



Hmm but that causes a problem for tboolean-box-a loadEvent in OpticksEventCompareTest::

    [blyth@localhost tmp]$ tboolean-box-a
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7
    Copyright (C) 2013 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-redhat-linux-gnu".
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>...
    Reading symbols from /home/blyth/local/opticks/lib/OpticksEventCompareTest...done.
    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/OpticksEventCompareTest --torch --tag 1 --cat tboolean-box --dbgnode 0 --dbgseqhis 0x8bd
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2019-06-02 22:36:10.514 FATAL [48764] [Opticks::configure@1717]  --interop mode with no cvd specified, adopting OPTICKS_DEFAULT_INTEROP_CVD hinted by envvar [1]
    2019-06-02 22:36:10.514 INFO  [48764] [Opticks::configure@1724]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    2019-06-02 22:36:10.526 ERROR [48764] [OpticksResource::initRunResultsDir@260] /home/blyth/local/opticks/results/OpticksEventCompareTest/R0_cvd_1/20190602_223610

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7b0b5f0 in OpticksEventSpec::getITag (this=0x0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:80
    80      return m_itag ; 
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff7b0b5f0 in OpticksEventSpec::getITag (this=0x0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:80
    #1  0x00007ffff7b0b3e2 in OpticksEventSpec::clone (this=0x0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:53
    #2  0x00007ffff7b14f03 in OpticksEvent::make (spec=0x0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEvent.cc:122
    #3  0x00007ffff7b38fe2 in Opticks::loadEvent (this=0x7fffffffd3c0, ok=true, tagoffset=0) at /home/blyth/opticks/optickscore/Opticks.cc:2212
    #4  0x0000000000403770 in main (argc=10, argv=0x7fffffffd9d8) at /home/blyth/opticks/optickscore/tests/OpticksEventCompareTest.cc:30
    (gdb) 


