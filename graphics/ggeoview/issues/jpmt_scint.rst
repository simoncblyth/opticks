JPMT Scint not working
=======================


::

    [2016-Mar-20 11:05:31.673557]:info: App:: uploadGeometryViz
    LoadArrayFromNumpy failed to open path /usr/local/env/opticks/juno/idomscintillation/1.npy 
    NPY<T>::load failed for path [/usr/local/env/opticks/juno/idomscintillation/1.npy] use debugload to see why
    [2016-Mar-20 11:05:31.673876]:warning: NumpyEvt::load NO SUCH EVENT : RUN WITHOUT --load OPTION TO CREATE IT  typ: scintillation tag: 1 det: juno cat:  udet: juno
    [2016-Mar-20 11:05:31.674010]:warning: App::loadEvtFromFile LOAD FAILED 
    /Users/blyth/env/graphics/ggeoview/ggeoview.bash: line 2023: 47543 Segmentation fault: 11  $bin $*
    bogon:opticks blyth$ 


Running with save required upping rng max, but runs into device memory limit::

    delta:npy blyth$ ggv-;ggv-jpmt-propagate-scintillation

    ggv-jpmt-propagate-scintillation () 
    { 
        ggv --jpmt --scintillation --compute --timemax 400 --animtimemax 200 --save
    }

    ...
    [2016-Mar-20 16:01:05.244775]:info: OConfig::getNumEntryPoint m_raygen_index 1 m_exception_index 1
    [2016-Mar-20 16:01:05.244861]:info: OContext::close numEntryPoint 1
    [2016-Mar-20 16:01:05.244967]:info: OContext::close m_raygen_index 1 m_exception_index 1
    OProg R 0 generate.cu.ptx generate 
    OProg E 0 generate.cu.ptx exception 
    [2016-Mar-20 16:01:05.361087]:info: OContext::launch entry 0 width 1493444 height 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Insufficient device memory. GPU does not support paging., [16515528])
    /Users/blyth/env/graphics/ggeoview/ggeoview.bash: line 2028: 66057 Abort trap: 6           $bin $*


::

    In [1]: s = np.load("scintillation/1.npy")

    In [2]: s.shape
    Out[2]: (1774, 6, 4)

    In [8]: np.save("scintillation/1_mod10.npy", s[::10])

    In [9]: np.save("scintillation/1_mod100.npy", s[::100])

    In [10]: np.load("scintillation/1_mod100.npy").shape
    Out[10]: (18, 6, 4)

    In [11]: np.save("scintillation/1_mod1000.npy",s[::1000])



::

    In [1]: i = np.load("/usr/local/env/opticks/juno/idomcerenkov/1.npy")

    In [2]: i
    Out[2]: array([[[ 0,  0,  0, 10]]], dtype=int32)

::

    delta:juno blyth$ l */*.npy
    -rw-r--r--  1 blyth  staff        128 Jan  4 18:41 fdomcerenkov/1.npy
    -rw-r--r--  1 blyth  staff         96 Jan  4 18:41 idomcerenkov/1.npy
    -rw-r--r--  1 blyth  staff   15700208 Jan  4 18:41 phcerenkov/1.npy
    -rw-r--r--  1 blyth  staff    3925112 Jan  4 18:41 pscerenkov/1.npy
    -rw-r--r--  1 blyth  staff   39250400 Jan  4 18:41 rscerenkov/1.npy
    -rw-r--r--  1 blyth  staff  157001360 Jan  4 18:41 rxcerenkov/1.npy
    -rw-r--r--  1 blyth  staff   62800592 Jan  4 18:41 oxcerenkov/1.npy

    -rw-r--r--  1 blyth  staff   33225232 Jul 26  2015 phscintillation/2.npy
    -rw-r--r--  1 blyth  staff  332251616 Jul 26  2015 rxscintillation/2.npy
    -rw-r--r--  1 blyth  staff  132900688 Jul 26  2015 oxscintillation/2.npy
    -rw-r--r--  1 blyth  staff     340592 Jul 26  2015 scintillation/2.npy
    -rw-r--r--  1 blyth  staff   23895184 Jul 26  2015 phscintillation/1.npy
    -rw-r--r--  1 blyth  staff  238951136 Jul 26  2015 rxscintillation/1.npy
    -rw-r--r--  1 blyth  staff   95580496 Jul 26  2015 oxscintillation/1.npy
    -rw-r--r--  1 blyth  staff     170384 Jul 25  2015 scintillation/1.npy
    -rw-r--r--  1 blyth  staff   62800592 Jul 24  2015 oxcerenkov/b.npy
    -rw-r--r--  1 blyth  staff   62800592 Jul 24  2015 oxcerenkov/a.npy
    -rw-r--r--  1 blyth  staff     368720 Jul 22  2015 cerenkov/1.npy



Running with ::

    [2016-Mar-20 16:24:16.993715]:info: OContext::close m_raygen_index 1 m_exception_index 1
    OProg R 0 generate.cu.ptx generate 
    OProg E 0 generate.cu.ptx exception 
    [2016-Mar-20 16:24:17.110304]:info: OContext::launch entry 0 width 981258 height 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Insufficient device memory. GPU does not support paging., [16515528])
    /Users/blyth/env/graphics/ggeoview/ggeoview.bash: line 2028: 70276 Abort trap: 6           $bin $*
    delta:cerenkov blyth$ 



