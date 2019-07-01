test-fails-from-geometry-workflow-interference
===================================================

Reported by Sam, https://groups.io/g/opticks/message/82


interpolationTest runs some python analysis which is failing for lack of OPTICKS_KEY envvar
----------------------------------------------------------------------------------------------

Reproduce with::

   unset OPTICKS_KEY ; unset IDPATH ; ipython --pdb -i /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py


The fork betwen legacy and direct init in ana/env.py is based on IDPATH being in environment::

    102     def __init__(self, legacy=False):
    103         self.ext = {}
    104         self.env = {}
    105 
    106         if os.environ.has_key("IDPATH"):
    107             self.legacy_init()
    108         else:
    109             self.direct_init()
    110         pass
    111 

Hmm yes, the C++ interpolationTest knows what IDPATH should be, so it should set the IDPATH envvar 
so the python script takes the legacy route : this is consistent with aim of making IDPATH an 
internal thing (not a user input).

::

     413 void GGeo::init()
     414 {
     415     LOG(verbose) << "[" ;
     416     const char* idpath = m_ok->getIdPath() ;
     417     LOG(verbose) << " idpath " << ( idpath ? idpath : "NULL" ) ;
     418     assert(idpath && "GGeo::init idpath is required" );
     419 
     420     bool cache_exists = m_ok->hasGeoCache();
     421     bool cache_requested = m_ok->isGeocache() ;
     422     m_loaded_from_cache = cache_exists && cache_requested ;
     423     bool will_load_libs = m_loaded_from_cache && !m_live ;
     424 
     425     LOG(error)
     426         << " idpath " << idpath
     427         << " cache_exists " << cache_exists
     428         << " cache_requested " << cache_requested
     429         << " m_loaded_from_cache " << m_loaded_from_cache
     430         << " m_live " << m_live
     431         << " will_load_libs " << will_load_libs
     432         ;

::

     585 void Opticks::initResource()
     586 {
     587     LOG(LEVEL) << "( OpticksResource " ;
     588     m_resource = new OpticksResource(this);
     589     LOG(LEVEL) << ") OpticksResource " ;
     590     setDetector( m_resource->getDetector() );
     591 
     592     const char* idpath = m_resource->getIdPath();
     593     m_parameters->add<std::string>("idpath", idpath);
     594 
     595     LOG(LEVEL) << " DONE " << m_resource->desc()  ;
     596 }



Turning up the volume, GBndLib::saveAllOverride is not saving to the intended $TMP/interpolationTest::

   GBndLib=ERROR GPropertyLib=ERROR interpolationTest 

Not honouring the override::

    1020 void GBndLib::saveAllOverride(const char* dir)
    1021 {
    1022     LOG(LEVEL) << "[ " << dir ;
    1023 
    1024     m_ok->setIdPathOverride(dir);
    1025 
    1026     save();             // only saves the guint4 bnd index
    1027     saveToCache();      // save float buffer too for comparison with wavelength.npy from GBoundaryLib with GBndLibTest.npy 
    1028     saveOpticalBuffer();
    1029 
    1030     m_ok->setIdPathOverride(NULL);
    1031 
    1032     LOG(LEVEL) << "]" ;
    1033 
    1034 }


Check with strace::

    strace -o /tmp/strace.log -e open interpolationTest 

Yep confirmed, thats dangerous and it would fail from permissions in a shared install::

    [blyth@localhost ggeo]$ strace.py 
    strace.py
     interpolationTest.log                                                            :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/OptixCache/cache.db                                                     :            O_RDWR|O_CREAT :  0666 
     /var/tmp/OptixCache/cache.db                                                     : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/OptixCache/cache.db-wal                                                 :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-shm                                                 :            O_RDWR|O_CREAT :  0664 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBndLib/GBndLibIndex.npy :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBndLib/GBndLib.npy :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/interpolationTest/GItemList/GBndLib.txt                       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBndLib/GBndLibOptical.npy :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/interpolationTest/interpolationTest_interpol.npy              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    [blyth@localhost ggeo]$ 



::

    470 void GPropertyLib::saveToCache(NPYBase* buffer, const char* suffix)
    471 {
    472     assert(suffix);
    473     std::string dir = getCacheDir();
    474     std::string name = getBufferName(suffix);
    475 
    476 

    368 std::string GPropertyLib::getCacheDir()
    369 {
    370     return m_resource->getPropertyLibDir(m_type);
    371 }


    834 void BOpticksResource::setIdPathOverride(const char* idpath_tmp)  // used for test saves into non-standard locations
    835 {
    836    m_idpath_tmp = idpath_tmp ? strdup(idpath_tmp) : NULL ;
    837 }
    838 const char* BOpticksResource::getIdPath() const
    839 {
    840     LOG(verbose) << "getIdPath"
    841               << " idpath_tmp " << m_idpath_tmp
    842               << " idpath " << m_idpath
    843               ;
    844 
    845     return m_idpath_tmp ? m_idpath_tmp : m_idpath  ;
    846 }



After fix in BOpticksResource::getPropertyLibDir are writing to intended override dir::

    [blyth@localhost ggeo]$ strace.py
    strace.py
     interpolationTest.log                                                            :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/OptixCache/cache.db                                                     :            O_RDWR|O_CREAT :  0666 
     /var/tmp/OptixCache/cache.db                                                     : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/OptixCache/cache.db-wal                                                 :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-shm                                                 :            O_RDWR|O_CREAT :  0664 
     /tmp/blyth/opticks/interpolationTest/GBndLib/GBndLibIndex.npy                    :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/interpolationTest/GBndLib/GBndLib.npy                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/interpolationTest/GItemList/GBndLib.txt                       :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/interpolationTest/GBndLib/GBndLibOptical.npy                  :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/interpolationTest/interpolationTest_interpol.npy              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    [blyth@localhost ggeo]$ 


Also fixed up interpolationTest : it had some stale paths.




integration tests assuming direct workflow when users dont have that setup ?
---------------------------------------------------------------------------------

* have removed the geocache-key-export from tboolean.sh it is more appropriate 
  for base geometry to be setup in users .bashrc










