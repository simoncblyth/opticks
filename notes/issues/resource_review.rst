resource_review
==================

ISSUE
------

Resource rejig migrating from layout 0 to layout 1 
splits geometry sources from the derived geocache.

Initially used kludge envvars as expedient ways to 
make the migration.

But juggling envvars is a kludge, and support nightmare, 
so trying to reduce.


APPROACH
---------

Used the known layouts to convert IDPATH into SRCPATH in 

* brap/BPath.cc brap/BOpticksResource.cc  see BPathTest 
* base/bpath.py
* opticks.bash:opticks-idpath2srcpath



TODO : avoid OInterpolationTest casualty
------------------------------------------


::

    2017-12-05 12:21:49.718 ERROR [257952] [*GBndLib::createBufferForTex2d@677] GBndLib::createBufferForTex2d mat 0x7fc5e355b8e0 sur 0x7fc5e355e280
    2017-12-05 12:21:49.720 INFO  [257952] [SLog::operator@15] OScene::OScene DONE
    2017-12-05 12:21:49.720 INFO  [257952] [main@131]  ok 
    2017-12-05 12:21:49.720 INFO  [257952] [GBndLib::saveAllOverride@896] GBndLib::saveAllOverride
    2017-12-05 12:21:49.720 ERROR [257952] [GPropertyLib::saveToCache@434] GPropertyLib::saveToCache dir /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GBndLib name GBndLibIndex.npy type GBndLib
    2017-12-05 12:21:49.720 ERROR [257952] [*GBndLib::createBufferForTex2d@677] GBndLib::createBufferForTex2d mat 0x7fc5e355b8e0 sur 0x7fc5e355e280
    2017-12-05 12:21:49.722 INFO  [257952] [GPropertyLib::close@409] GPropertyLib::close type GBndLib buf 123,4,2,39,4
    2017-12-05 12:21:49.724 ERROR [257952] [GPropertyLib::saveToCache@434] GPropertyLib::saveToCache dir /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GBndLib name GBndLibOptical.npy type GBndLib
    2017-12-05 12:21:49.724 INFO  [257952] [OInterpolationTest::launch@87] OInterpolationTest::launch nb   123 nx   761 ny   984 progname             OInterpolationTest name OInterpolationTest_interpol.npy base $TMP/InterpolationTest
    2017-12-05 12:21:49.724 INFO  [257952] [OLaunchTest::init@50] OLaunchTest entry   0 width       1 height       1 ptx                          OInterpolationTest.cu.ptx prog                                 OInterpolationTest
    2017-12-05 12:21:49.724 INFO  [257952] [OLaunchTest::launch@61] OLaunchTest entry   0 width     761 height     123 ptx                          OInterpolationTest.cu.ptx prog                                 OInterpolationTest
    2017-12-05 12:21:49.724 INFO  [257952] [OContext::close@236] OContext::close numEntryPoint 1
    2017-12-05 12:21:49.724 INFO  [257952] [OContext::close@240] OContext::close setEntryPointCount done.
    2017-12-05 12:21:49.737 INFO  [257952] [OContext::close@246] OContext::close m_cfg->apply() done.
    2017-12-05 12:21:53.126 INFO  [257952] [OContext::launch@323] OContext::launch LAUNCH time: 3.38909
    args: /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py
    Traceback (most recent call last):
      File "/Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py", line 17, in <module>
        blib = PropLib.load_GBndLib(base)
      File "/Users/blyth/opticks/ana/proplib.py", line 96, in load_GBndLib
        t = np.load(os.path.expandvars(os.path.join(base,"GBndLib/GBndLib.npy")))
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/lib/npyio.py", line 369, in load
        fid = open(file, "rb")
    IOError: [Errno 2] No such file or directory: '/tmp/blyth/opticks/InterpolationTest/GBndLib/GBndLib.npy'
    2017-12-05 12:21:53.281 INFO  [257952] [SSys::run@46] python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    2017-12-05 12:21:53.281 WARN  [257952] [SSys::run@53] SSys::run FAILED with  cmd python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py
    simon:opticks blyth$ 




TODO : eliminate envvar OPTICKS_RESOURCE_LAYOUT
------------------------------------------------

Initially::

    simon:~ blyth$ opticks-find OPTICKS_RESOURCE_LAYOUT 
    ./boostrap/BOpticksResource.cc:    m_layout(SSys::getenvint("OPTICKS_RESOURCE_LAYOUT", 0)),
    ./boostrap/BOpticksResource.cc:    addName("OPTICKS_RESOURCE_LAYOUT", layout );
    ./ana/base.py:        #self.layout = int(os.environ.get("OPTICKS_RESOURCE_LAYOUT", 0))
    simon:opticks blyth$ 


Src side is same in layout 0,1 so cannot extract from SRCPATH. 
Hmm have to just fix it to 1. 

This means can only get rid of the envvar once ready to fix to layout 1.

* python always needs the IDPATH, so it can detect the layout 


DONE : eliminate OPTICKS_SRCPATH 
------------------------------------------------------------------


Finally::

    simon:boostrap blyth$ opticks-find OPTICKS_SRCPATH
    ./boostrap/tests/BOpticksResourceTest.cc:    const char* srcpath   = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    ./opticksnpy/tests/NSceneTest.cc:    const char* srcpath = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    simon:opticks blyth$ 


Initially::

    simon:boostrap blyth$ opticks-find SRCPATH
    ./opticks.bash:opticks-srcfold(){ echo $(dirname $OPTICKS_SRCPATH) ; }
    ./opticks.bash:  OPTICKS_SRCPATH          : $OPTICKS_SRCPATH 
    ./boostrap/BOpticksResource.cc:    const char* srcpath = SSys::getenvvar("OPTICKS_SRCPATH"); 
    ./boostrap/tests/BOpticksResourceTest.cc:    const char* srcpath   = SSys::getenvvar("OPTICKS_SRCPATH");
    ./boostrap/tests/BOpticksResourceTest.cc:                  << " OPTICKS_SRCPATH " << srcpath 
    ./opticksnpy/tests/HitsNPYTest.cc:        printf("%s : requires OPTICKS_SRCPATH  envvar \n", argv[0]);
    ./opticksnpy/tests/NSceneTest.cc:    const char* srcpath = SSys::getenvvar("OPTICKS_SRCPATH");
    ./opticksnpy/tests/NSensorListTest.cc:        printf("%s : requires OPTICKS_SRCPATH  envvar \n", argv[0]);
    ./boostrap/BOpticksResource.hh:       static const char* IdMapSrcPath(); // requires OPTICKS_SRCPATH  envvar
    ./ana/base.py:OPTICKS_SRCPATH = os.path.expandvars("$OPTICKS_SRCPATH")
    ./ana/base.py:            if OPTICKS_SRCPATH == "$OPTICKS_SRCPATH":
    ./ana/base.py:                print "ana/base.py:OpticksEnv missing OPTICKS_SRCPATH envvar [%s] (full path to .dae geometry file) " % OPTICKS_SRCPATH
    ./ana/base.py:            if not os.path.isfile(OPTICKS_SRCPATH): 
    ./ana/base.py:                print "ana/base.py:OpticksEnv warning OPTICKS_SRCPATH file does not exist [%s] " % OPTICKS_SRCPATH
    ./ana/base.py:            self.srcpath = OPTICKS_SRCPATH 
    simon:opticks blyth$ 
    simon:opticks blyth$ 

    simon:ana blyth$ opticks-find SRCPATH
    ./opticks.bash:opticks-srcfold(){ echo $(dirname $OPTICKS_SRCPATH) ; }
    ./opticks.bash:  OPTICKS_SRCPATH          : $OPTICKS_SRCPATH 

    ./boostrap/BOpticksResource.cc:    const char* srcpath = SSys::getenvvar("OPTICKS_SRCPATH"); 
    ./boostrap/tests/BOpticksResourceTest.cc:    const char* srcpath   = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");

    ./opticksnpy/tests/HitsNPYTest.cc:        printf("%s : requires OPTICKS_SRCPATH  envvar \n", argv[0]);
    ./opticksnpy/tests/NSceneTest.cc:    const char* srcpath = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    ./opticksnpy/tests/NSensorListTest.cc:        printf("%s : requires OPTICKS_SRCPATH  envvar \n", argv[0]);
    ./boostrap/BOpticksResource.hh:       static const char* IdMapSrcPath(); // requires OPTICKS_SRCPATH  envvar



Got did of this, with BPath::

    267 const char* BOpticksResource::IdMapSrcPath()
    268 {
    269     const char* srcpath = SSys::getenvvar("OPTICKS_SRCPATH");
    270     return srcpath ? MakeSrcPath( srcpath, ".idmap" ) : NULL ;
    271 
    272 }

    simon:opticksnpy blyth$ opticks-find IdMapSrcPath
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::IdMapSrcPath()
    ./opticksnpy/tests/HitsNPYTest.cc:    const char* idmpath = BOpticksResource::IdMapSrcPath(); 
    ./opticksnpy/tests/NSensorListTest.cc:    const char* idmpath = BOpticksResource::IdMapSrcPath(); 
    ./boostrap/BOpticksResource.hh:       static const char* IdMapSrcPath(); // requires OPTICKS_SRCPATH  envvar
    simon:opticks blyth$ 



Now for bash::

    simon:opticks blyth$ opticks-find OPTICKS_SRCPATH
    ./opticks.bash:opticks-srcfold(){ echo $(dirname $OPTICKS_SRCPATH) ; }
    ./opticks.bash:  OPTICKS_SRCPATH          : $OPTICKS_SRCPATH 
    ./boostrap/tests/BOpticksResourceTest.cc:    const char* srcpath   = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    ./opticksnpy/tests/NSceneTest.cc:    const char* srcpath = SSys::getenvvar("DEBUG_OPTICKS_SRCPATH");
    simon:opticks blyth$ 

See bash functions::

    opticks-paths
    opticks-idpath2srcpath
    opticks-idpath2srcpath-test



