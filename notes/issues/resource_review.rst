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


TODO : eliminate envvar OPTICKS_RESOURCE_LAYOUT
------------------------------------------------

Initially::

    simon:~ blyth$ opticks-find OPTICKS_RESOURCE_LAYOUT 
    ./boostrap/BOpticksResource.cc:    m_layout(SSys::getenvint("OPTICKS_RESOURCE_LAYOUT", 0)),
    ./boostrap/BOpticksResource.cc:    addName("OPTICKS_RESOURCE_LAYOUT", layout );
    ./ana/base.py:        #self.layout = int(os.environ.get("OPTICKS_RESOURCE_LAYOUT", 0))
    simon:opticks blyth$ 



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



