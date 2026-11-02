Inadvertent_SCurandState_use_revealed_by_sudo_ncu_CSGOptiXSMTest_running
=========================================================================

Overview
-----------

ncu running under sudo reveals a hangover from before
switching to Philox - that avoids the need for CurandState files.

HMM: do to avoid this setup for Philox usage ? 


Fixed this issue mainly in QRng.hh QRng.cc
---------------------------------------------

* fixed by use of the below macros to cut away non-needed state handling::

    RNG_XORWOW
    RNG_PHILOX


Where is the Philox switch ?  RNG_PHILOX
------------------------------------------

sysrap/CMakeLists.txt RNG_PHILOX::

     908 set(SYSRAP_COMPILE_DEFINITIONS
     909       $<$<CONFIG:Debug>:CONFIG_Debug>
     910       $<$<CONFIG:RelWithDebInfo>:CONFIG_RelWithDebInfo>
     911       $<$<CONFIG:Release>:CONFIG_Release>
     912       $<$<CONFIG:MinSizeRel>:CONFIG_MinSizeRel>
     913 
     914       OPTICKS_SYSRAP
     915       WITH_CHILD
     916       PLOG_LOCAL
     917       RNG_PHILOX
     918 
     919       $<$<CONFIG:Debug>:DEBUG_PIDX>
     920       $<$<CONFIG:Debug>:DEBUG_PIDXYZ>
     921       $<$<CONFIG:Release>:PRODUCTION>
     922 )
     923 



Error from non-existing (for root) $HOME/.opticks/rngcache/RNG
----------------------------------------------------------------

ncu running under sudo reveals unexpected attempt to OPEN $HOME/.opticks/rngcache/RNG:: 

    [sudo] password for blyth: 
    ==PROF== Connected to process 2938103 (/tmp/blyth/local/opticks_Release_g411/lib/CSGOptiXSMTest)
    sdirectory::DirList FAILED TO OPEN DIR /root/.opticks/rngcache/RNG
    SGLM::initView VIEW [-] load_interpolated_view NO  interpolated_view.brief -
    ==PROF== Profiling "optixAccelBuild" - 0: 0%....50%....100% - 14 passes
    =


Thats from SCurandChunk.h use of DirList::


    [lo] A[blyth@localhost nv_profiling]$ opticks-f DirList
    ...
    ./sysrap/SCurandChunk.h:    sdirectory::DirList( names, dir, PREFIX, EXT ); 


Add ABORT at the error "sdirectory::DirList FAILED TO OPEN DIR /root/.opticks/rngcache/RNG" to pin it down and remove it
--------------------------------------------------------------------------------------------------------------------------

Inadvertent use of SCurandState - when there is no need to use that with RNG_PHILOX::

    New Thread 0x7fffbbdff000 (LWP 2989650)]
    sdirectory::DirList FAILED TO OPEN DIR /root/.opticks/rngcache/RNG
    CSGOptiXSMTest: /tmp/blyth/local/opticks_Debug_g411/include/SysRap/sdirectory.h:81: static void sdirectory::DirList(std::vector<std::__cxx11::basic_string<char> >&, const char*, const char*, const char*): Assertion `dir' failed.

    Thread 1 "CSGOptiXSMTest" received signal SIGABRT, Aborted.
    0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 nvidia-driver-cuda-libs-610.43.02-1.el9.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff483eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff482875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff4837886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff6148583 in sdirectory::DirList (names=..., path=0x16a86740 "/root/.opticks/rngcache/RNG", pfx=0x7ffff6231bf1 "SCurandChunk_", ext=0x7ffff6231bec ".bin") at /tmp/blyth/local/opticks_Debug_g411/include/SysRap/sdirectory.h:81
    #6  0x00007ffff6148f42 in SCurandChunk::ParseDir (chunk=..., _dir=0x0) at /tmp/blyth/local/opticks_Debug_g411/include/SysRap/SCurandChunk.h:143
    #7  0x00007ffff614acb3 in SCurandState::init (this=0x16a86428) at /tmp/blyth/local/opticks_Debug_g411/include/SysRap/SCurandState.h:117
    #8  0x00007ffff614abb5 in SCurandState::SCurandState (this=0x16a86428, _dir=0x0) at /tmp/blyth/local/opticks_Debug_g411/include/SysRap/SCurandState.h:86
    #9  0x00007ffff6146567 in QRng::QRng (this=0x16a863d0, skipahead_event_offset_=100000) at /home/blyth/opticks/qudarap/QRng.cc:64
    #10 0x00007ffff6085aab in QSim::UploadComponents (ssim=0x44ec00) at /home/blyth/opticks/qudarap/QSim.cc:137
    #11 0x00007ffff7e5acf0 in CSGOptiX::InitSim (ssim=0x44ec00) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:272
    #12 0x00007ffff7e5b3c9 in CSGOptiX::Create (fd=0x451300) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:357
    #13 0x00007ffff7e5a7aa in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:190
    #14 0x0000000000404933 in main (argc=1, argv=0x7fffffffba18) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 





::

     41 /**
     42 QRng::QRng
     43 ------------
     44 
     45 QRng instanciation is invoked from QSim::UploadComponents
     46 
     47 **/
     48 
     49 QRng::QRng(unsigned skipahead_event_offset_)
     50     :
     51     RNGNAME(srng<RNG>::NAME),
     52     UPLOAD_RNG_STATES(srng<RNG>::UPLOAD_RNG_STATES),
     53     skipahead_event_offset(skipahead_event_offset_),
     54     seed(0ull),
     55     offset(0ull),
     56     SEED_OFFSET(ssys::getenvvar("QRng__SEED_OFFSET")),
     57     parse_rc(SCurandSpec::ParseSeedOffset(seed, offset, SEED_OFFSET )),
     58     qr(new qrng<RNG>(seed, offset, skipahead_event_offset)),
     59     d_qr(nullptr),
     60 #ifdef OLD_MONOLITHIC_CURANDSTATE
     61     rngmax(0)
     62 #else
     63     rngmax(SEventConfig::MaxCurand()),
     64     cs(nullptr)
     ^^^^^^^^^^^^^^^^^^^^^^^^^
     65 #endif
     66 {
     67     init();
     68 }
     69 






::

     36 struct QUDARAP_API QRng
     37 {   
     38     typedef unsigned long long ULL ;
     39     static constexpr const char* QRng__LIFECYCLE = "QRng__LIFECYCLE" ;
     40     static bool LIFECYCLE ;
     41 
     42     static constexpr const ULL M = 1000000 ;
     43     static const plog::Severity LEVEL ;
     44     static const QRng* INSTANCE ;
     45     static const QRng* Get();
     46     static std::string Desc();
     47 
     48     static const char* Load_FAIL_NOTES ;
     49 
     50     
     51     
     52 #ifdef OLD_MONOLITHIC_CURANDSTATE
     53     static constexpr const char* IMPL = "OLD_MONOLITHIC_CURANDSTATE" ;
     54     static XORWOW* LoadAndUpload(ULL& rngmax, const char* path);
     55     static XORWOW* Load(ULL& rngmax, const char* path);
     56     static XORWOW* UploadAndFree(RNG* h_states, ULL num_states );
     57 #else
     58     static constexpr const char* IMPL = "CHUNKED_CURANDSTATE" ;
     59     static XORWOW* LoadAndUpload(ULL rngmax, const SCurandState& cs);
     60 #endif
     61     static void Save( XORWOW* states, unsigned num_states, const char* path );
     62 
     63 
     64     const char* RNGNAME ;
     65     bool  UPLOAD_RNG_STATES ;
     66     ULL            skipahead_event_offset ;
     67     ULL            seed ;
     68     ULL            offset ;
     69     const char*    SEED_OFFSET ;
     70     int            parse_rc ;
     71     qrng<RNG>*     qr ;
     72     qrng<RNG>*     d_qr ;
     73     ULL            rngmax ;
     74 
     75 #ifndef OLD_MONOLITHIC_CURANDSTATE
     76     SCurandState   cs ;
     77 #endif
     78 
     79 
     80     QRng(unsigned skipahead_event_offset=1) ;
     81     void init();
     82 
     83     template<typename R> void initStates();
     84 
     85     void initMeta();
     86 
     87     virtual ~QRng();
     88 
     89     std::string desc() const ;
     90 






::

    103 inline void SCurandState::init()
    104 {
    105     all = {} ;
    106     all.chunk_idx = 0 ;
    107     all.chunk_offset = 0 ;
    108     all.num = 0 ;
    109     all.states = nullptr ;
    110 
    111     const char* SEED_OFFSET = ssys::getenvvar("SCurandState__init_SEED_OFFSET");
    112     const char* CHUNKSIZES  = ssys::getenvvar("SCurandState__init_CHUNKSIZES");
    113 
    114     SCurandSpec::ParseSeedOffset( all.seed, all.offset, SEED_OFFSET );
    115     SCurandSpec::ParseChunkSizes( spec, CHUNKSIZES  );
    116 
    117     SCurandChunk::ParseDir(chunk, dir);
    118 
    119     int num_spec = spec.size() ;
    120     int num_chunk = chunk.size() ;
    121 
    122     if(num_chunk == 0)   // no chunk files found
    123     {
    124         initFromSpec();
    125     }
    126     else if( num_chunk > 0 )
    127     {
    128         initFromDir();
    129     }
    130 
    131     all.num = num_total();
    132 
    133     assert( spec.size() == chunk.size() );
    134 }





::

    028 struct SYSRAP_API SCurandChunk
     29 {
     30     typedef unsigned long long ULL ;
     31     scurandref<curandStateXORWOW> ref = {} ;
     32 
     33     std::string desc() const ;
     34     std::string meta() const ;
     35     std::string name() const ;
     36     const char* path(const char* _dir=nullptr) const ;
     37     bool path_exists(const char* _dir=nullptr) const ;
     38 
     39     static constexpr const long STATE_SIZE = 44 ;
     40     static constexpr const char* RNGDIR = "${RNGDir:-$HOME/.opticks/rngcache/RNG}" ;
     41     static constexpr const ULL M = 1000000 ;
     42     static constexpr const char* PREFIX = "SCurandChunk_" ;
     43     static constexpr const char* EXT = ".bin" ;
     44     static constexpr char DELIM = '_' ;




    130 /**
    131 SCurandChunk::ParseDir
    132 -----------------------
    133 
    134 Populate chunk vector based on matching file names within directory 
    135 
    136 **/
    137 
    138 
    139 inline int SCurandChunk::ParseDir(std::vector<SCurandChunk>& chunk, const char* _dir )
    140 {
    141     const char* dir = spath::Resolve(_dir ? _dir : RNGDIR) ;
    142     std::vector<std::string> names ;
    143     sdirectory::DirList( names, dir, PREFIX, EXT );
    144 
    145     int num_names = names.size();
    146     int count = 0 ;
    147 
    148     for(int i=0 ; i < num_names ; i++)
    149     {
    150         const std::string& n = names[i] ;
    151         SCurandChunk c = {} ;
    152         if(SCurandChunk::ParseName(c, n.c_str())==0)
    153         {
    154             ULL chunk_offset = NumTotal_InRange(chunk, 0, chunk.size() );
    155             assert( c.ref.chunk_offset == chunk_offset );
    156             assert( c.ref.chunk_idx == count ); // chunk files must be in idx order 
    157             chunk.push_back(c);
    158             count += 1 ;
    159         }
    160     }
    161     return 0 ;
    162 }



