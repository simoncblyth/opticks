static_initializer_backtrace
==============================


Interesting backtrace from an error in a static initializer::


    (lldb) r
    Process 97801 launched: '/usr/local/opticks/lib/U4PMTFastSimTest' (x86_64)
    SLOG::EnvLevel adjusting loglevel by envvar   key SEventConfig level INFO fallback DEBUG upper_level INFO
    Process 97801 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
        frame #0: 0x00007fff70312232 libsystem_c.dylib`strlen + 18
    libsystem_c.dylib`strlen:
    ->  0x7fff70312232 <+18>: pcmpeqb (%rdi), %xmm0
        0x7fff70312236 <+22>: pmovmskb %xmm0, %esi
        0x7fff7031223a <+26>: andq   $0xf, %rcx
        0x7fff7031223e <+30>: orq    $-0x1, %rax
    Target 0: (U4PMTFastSimTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x00007fff70312232 libsystem_c.dylib`strlen + 18
        frame #1: 0x0000000106137e90 libSysRap.dylib`U::EndsWith(s=0x0000000000000000, q=".npy") at NPU.hh:490
        frame #2: 0x00000001061bff28 libSysRap.dylib`NP::Load(path_="$SEvt__UU_BURN") at NP.hh:4016
        frame #3: 0x0000000106247210 libSysRap.dylib`::__cxx_global_var_init() at SEvt.cc:36
        frame #4: 0x00000001062472c9 libSysRap.dylib`_GLOBAL__sub_I_SEvt.cc at SEvt.cc:0
        frame #5: 0x00000001000eaac6 dyld`ImageLoaderMachO::doModInitFunctions(ImageLoader::LinkContext const&) + 420
        frame #6: 0x00000001000eacf6 dyld`ImageLoaderMachO::doInitialization(ImageLoader::LinkContext const&) + 40
        frame #7: 0x00000001000e6218 dyld`ImageLoader::recursiveInitialization(ImageLoader::LinkContext const&, unsigned int, char const*, ImageLoader::InitializerTimingList&, ImageLoader::UninitedUpwards&) + 330
        frame #8: 0x00000001000e61ab dyld`ImageLoader::recursiveInitialization(ImageLoader::LinkContext const&, unsigned int, char const*, ImageLoader::InitializerTimingList&, ImageLoader::UninitedUpwards&) + 221
        frame #9: 0x00000001000e61ab dyld`ImageLoader::recursiveInitialization(ImageLoader::LinkContext const&, unsigned int, char const*, ImageLoader::InitializerTimingList&, ImageLoader::UninitedUpwards&) + 221
        frame #10: 0x00000001000e61ab dyld`ImageLoader::recursiveInitialization(ImageLoader::LinkContext const&, unsigned int, char const*, ImageLoader::InitializerTimingList&, ImageLoader::UninitedUpwards&) + 221
        frame #11: 0x00000001000e534e dyld`ImageLoader::processInitializers(ImageLoader::LinkContext const&, unsigned int, ImageLoader::InitializerTimingList&, ImageLoader::UninitedUpwards&) + 134
        frame #12: 0x00000001000e53e2 dyld`ImageLoader::runInitializers(ImageLoader::LinkContext const&, ImageLoader::InitializerTimingList&) + 74
        frame #13: 0x00000001000d6567 dyld`dyld::initializeMainExecutable() + 196
        frame #14: 0x00000001000db239 dyld`dyld::_main(macho_header const*, unsigned long, int, char const**, char const**, char const**, unsigned long*) + 7242
        frame #15: 0x00000001000d53d4 dyld`dyldbootstrap::start(macho_header const*, int, char const**, long, macho_header const*, unsigned long*) + 453
        frame #16: 0x00000001000d51d2 dyld`_dyld_start + 54
    (lldb) 

    (lldb) f 3
    frame #3: 0x0000000106247210 libSysRap.dylib`::__cxx_global_var_init() at SEvt.cc:36
       33  	
       34  	
       35  	NP* SEvt::UU = nullptr ; 
    -> 36  	NP* SEvt::UU_BURN = NP::Load("$SEvt__UU_BURN") ; 
       37  	
       38  	const plog::Severity SEvt::LEVEL = SLOG::EnvLevel("SEvt", "DEBUG"); 
       39  	const int SEvt::GIDX = SSys::getenvint("GIDX",-1) ;
    (lldb) f 2
    frame #2: 0x00000001061bff28 libSysRap.dylib`NP::Load(path_="$SEvt__UU_BURN") at NP.hh:4016
       4013	            << std::endl 
       4014	            ; 
       4015	
    -> 4016	    bool npy_ext = U::EndsWith(path, EXT) ; 
       4017	    NP* a = nullptr ; 
       4018	    if(npy_ext)
       4019	    {
    (lldb) p path
    (const char *) $0 = 0x0000000000000000
    (lldb) 



