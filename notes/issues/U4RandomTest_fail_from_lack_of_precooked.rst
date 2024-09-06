U4RandomTest_fail_from_lack_of_precooked
============================================

Needs cleaner handling...

* eg use static creator and check ahead before instanciation for cleaning handling lack of precooked

::

    SLOG::EnvLevel adjusting loglevel by envvar   key U4Random level INFO fallback DEBUG upper_level INFO
    NP::load Failed to load from path /home/blyth/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy

    Program received signal SIGINT, Interrupt.
    __pthread_kill_implementation (threadid=<optimized out>, signo=signo@entry=2, no_tid=no_tid@entry=0) at pthread_kill.c:44
    44	      return INTERNAL_SYSCALL_ERROR_P (ret) ? INTERNAL_SYSCALL_ERRNO (ret) : 0;
    (gdb) bt
    #0  __pthread_kill_implementation (threadid=<optimized out>, signo=signo@entry=2, no_tid=no_tid@entry=0) at pthread_kill.c:44
    #1  0x00007ffff3aa15b3 in __pthread_kill_internal (signo=2, threadid=<optimized out>) at pthread_kill.c:78
    #2  0x00007ffff3a54d06 in __GI_raise (sig=2) at ../sysdeps/posix/raise.c:26
    #3  0x00007ffff7f345d8 in NP::load(char const*) () from /data1/blyth/local/opticks_Release/lib/../lib64/libU4.so
    #4  0x00007ffff7f349ab in NP::Load(char const*) () from /data1/blyth/local/opticks_Release/lib/../lib64/libU4.so
    #5  0x00007ffff7f6f907 in U4Random::U4Random(char const*, char const*) () from /data1/blyth/local/opticks_Release/lib/../lib64/libU4.so
    #6  0x00000000004026ed in main ()
    (gdb) f 3
    #3  0x00007ffff7f345d8 in NP::load(char const*) () from /data1/blyth/local/opticks_Release/lib/../lib64/libU4.so
    (gdb) f 5
    #5  0x00007ffff7f6f907 in U4Random::U4Random(char const*, char const*) () from /data1/blyth/local/opticks_Release/lib/../lib64/libU4.so
    (gdb) 



::

    4923 inline int NP::load(const char* _path)
    4924 {
    4925     nodata = NoData(_path) ;  // _path starting with '@' 
    4926     const char* path = nodata ? _path + 1 : _path ;
    4927 
    4928     if(VERBOSE) std::cerr << "[ NP::load " << path << std::endl ;
    4929 
    4930     lpath = path ;  // loadpath 
    4931     lfold = U::DirName(path);
    4932 
    4933     std::ifstream fp(path, std::ios::in|std::ios::binary);
    4934     if(fp.fail())
    4935     {
    4936         std::cerr << "NP::load Failed to load from path " << path << std::endl ;
    4937         std::raise(SIGINT);
    4938         return 1 ;
    4939     }


::

    083 U4Random::U4Random(const char* seq, const char* seqmask)
     84     :
     85     m_seqpath(SPath::Resolve( seq ? seq : SeqPath(), NOOP)),
     86     m_seq(m_seqpath ? NP::Load(m_seqpath) : nullptr),
     87     m_seq_values(m_seq ? m_seq->cvalues<float>() : nullptr ),
     88     m_seq_ni(m_seq ? m_seq->shape[0] : 0 ),                        // num items
     89     m_seq_nv(m_seq ? m_seq->shape[1]*m_seq->shape[2] : 0 ),        // num values in each item 
     90     m_seq_index(-1),
     91 
     92     m_cur(NP::Make<int>(m_seq_ni)),
     93     m_cur_values(m_cur->values<int>()),
     94     m_recycle(true),
     95     m_default(CLHEP::HepRandom::getTheEngine()),
     96 
     97     m_seqmask(seqmask ? NP::Load(seqmask) : nullptr),
     98     m_seqmask_ni( m_seqmask ? m_seqmask->shape[0] : 0 ),
     99     m_seqmask_values(m_seqmask ? m_seqmask->cvalues<size_t>() : nullptr),
    100     //m_flat_debug(SSys::getenvbool("U4Random_flat_debug")),
    101     m_flat_prior(0.),
    102     m_ready(false),
    103     m_select(SSys::getenvintvec("U4Random_select")),
    104     m_select_action(SDBG::Action(SSys::getenvvar("U4Random_select_action", "backtrace")))   // "backtrace" "caller" "interrupt" "summary"
    105 {
    106     init();
    107 }



Precooked dir not being resolved by spath::Resolve
------------------------------------------------------

::

    P[blyth@localhost sysrap]$ grep PrecookedDir *.*
    SOpticksResource.cc:const char* SOpticksResource::PrecookedDir(){   return SPath::Resolve(ResolvePrecookedPrefix(), "precooked", NOOP); }
    SOpticksResource.cc:    const char* precooked_dir = PrecookedDir() ; 
    SOpticksResource.cc:        << "SOpticksResource::PrecookedDir()           " << ( precooked_dir ? precooked_dir : "-" )  << std::endl 
    SOpticksResource.cc:const char* SOpticksResource::KEYS = "IDPath CFBase CFBaseAlt GeocacheDir RuncacheDir RNGDir PrecookedDir DefaultOutputDir SomeGDMLPath GDMLPath GEOMSub GEOMWrap CFBaseFromGEOM UserGEOMDir GEOMList" ; 
    SOpticksResource.cc:|   PrecookedDir          |                                                     |
    SOpticksResource.cc:    else if( strcmp(key, "PrecookedDir")==0)     tok = SOpticksResource::PrecookedDir(); 
    SOpticksResource.hh:    static const char* PrecookedDir();
    P[blyth@localhost sysrap]$ 





