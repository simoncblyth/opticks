revive_aligned_running
=======================

* :doc:`tboolean_box_perfect_alignment`


Had another go at this in 
---------------------------

* :doc:`tboolean_box_perfect_alignment_revisited`


Attempt to repeat some of *tboolean_box_perfect_alignment*
----------------------------------------------------------------------------


1. interop problem : Cannot get device pointers from non-CUDA interop buffers  : FIXED THIS A FEWS DAYS AGO WITH downloadHitsInterop 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the OptiX version bump to 5.1  are now getting this interop problem, both on macOS as well as Linux. 
Workaround is to add *--compute* to switch to non-interop non-viz mode and do a separate 
subsequent  *--load* when want to visualize.

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero
    ...
    2018-08-09 21:44:57.967 INFO  [11616473] [OpticksViz::downloadEvent@317] OpticksViz::downloadEvent (1) DONE 
    2018-08-09 21:44:57.967 INFO  [11616473] [OEvent::download@358] OEvent::download id 1
    2018-08-09 21:44:57.967 INFO  [11616473] [OContext::download@439] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    2018-08-09 21:44:57.969 ERROR [11616473] [OEvent::downloadHits@399] OEvent::downloadHits.cpho
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void **)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)



2. missing precooked RNG file at $TMP/TRngBufTest.npy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --compute

    ...

    2018-08-09 21:47:10.866 INFO  [11618668] [CRec::initEvent@82] CRec::initEvent note recstp
    HepRandomEngine::put called -- no effect!
    2018-08-09 21:47:11.189 INFO  [11618668] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2018-08-09 21:47:11.189 INFO  [11618668] [CInputPhotonSource::GeneratePrimaryVertex@163] CInputPhotonSource::GeneratePrimaryVertex n 10000
    2018-08-09 21:47:11.202 FATAL [11618668] [CRandomEngine::setupCurandSequence@124] CRandomEngine::setupCurandSequence m_curand_ni ZERO  no precooked RNG have been loaded from  m_path $TMP/TRngBufTest.npy : try running : TRngBufTest 
    Assertion failed: (m_curand_ni > 0), function setupCurandSequence, file /Users/blyth/opticks/cfg4/CRandomEngine.cc, line 132.
    /Users/blyth/opticks/bin/op.sh: line 846: 35959 Abort trap: 6           /usr/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --compute --rendermode +global,+axis --animtimemax 20 --timemax 20 --geocenter --stack 2180 --eye 1,0,0 --dbganalytic --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box--_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-box--_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 3 --cat tboolean-box --anakey tboolean --save
    /Users/blyth/opticks/bin/op.sh RC 134
    epsilon:analytic blyth$ 


::

    epsilon:analytic blyth$ ll /tmp/blyth/opticks/TRngBufTest.npy 
    -rw-r--r--  1 blyth  wheel  204800096 Aug  9 21:48 /tmp/blyth/opticks/TRngBufTest.npy
    epsilon:analytic blyth$ du -h /tmp/blyth/opticks/TRngBufTest.npy 
    195M	/tmp/blyth/opticks/TRngBufTest.npy
    epsilon:analytic blyth$ 



3. runs both simulations but issue at auto analysis stage 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After some iteration on the anakey python commandline, fixing some python formatting
succeed to run the analysis.  The perfect match is still seen.

::

   ipython -i -- /Users/blyth/opticks/ana/tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch


    epsilon:analytic blyth$ ipython -i -- /Users/blyth/opticks/ana/tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch
    args: /opt/local/bin/ipython -i -- /Users/blyth/opticks/ana/tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch
    [2018-08-09 22:10:17,122] p36319 {/Users/blyth/opticks/ana/base.py:339} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
    args: /Users/blyth/opticks/ana/tboolean.py --tag 3 --tagoffset 0 --det tboolean-box --src torch
    [2018-08-09 22:10:17,126] p36319 {/Users/blyth/opticks/ana/tboolean.py:62} INFO - tag 3 src torch det tboolean-box c2max 2.0 ipython True 
    [2018-08-09 22:10:17,126] p36319 {/Users/blyth/opticks/ana/ab.py:110} INFO - ab START
    [2018-08-09 22:10:17,280] p36319 {/Users/blyth/opticks/ana/seq.py:270} INFO -  c2sum 0.0 ndf 7 c2p 0.0 c2_pval 1 
    [2018-08-09 22:10:17,282] p36319 {/Users/blyth/opticks/ana/seq.py:270} INFO -  c2sum 0.0 ndf 6 c2p 0.0 c2_pval 1 
    ab.a.metadata:                 /tmp/blyth/opticks/evt/tboolean-box/torch/3 ca960875c829b9b5d2462c265fdb7510 c73dd7e7dad8c7e239794d2f2eda381c  100000    -1.0000 COMPUTE_MODE 
    [2018-08-09 22:10:17,287] p36319 {/Users/blyth/opticks/ana/seq.py:270} INFO -  c2sum 0.0 ndf 7 c2p 0.0 c2_pval 1 
    [2018-08-09 22:10:17,288] p36319 {/Users/blyth/opticks/ana/seq.py:270} INFO -  c2sum 0.0 ndf 6 c2p 0.0 c2_pval 1 
    [2018-08-09 22:10:17,290] p36319 {/Users/blyth/opticks/ana/seq.py:270} INFO -  c2sum 0.0 ndf 6 c2p 0.0 c2_pval 1 
    AB(3,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  3 :  20180809-2150 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/3/fdom.npy () 
    B tboolean-box/torch/ -3 :  20180809-2150 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-3/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  3:tboolean-box   -3:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO BR SA
    0002            8cbcd      5420      5420             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       349       349             0.00        1.000 +- 0.054        1.000 +- 0.054  [6 ] TO BT BR BR BT SA
    0004              86d        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO SC SA
    0005          8cbbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [7 ] TO BT BR BR BR BT SA
    0006            86ccd        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] TO BT BT SC SA
    0007              4cd        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] TO BT AB
    0008       bbbbbbb6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [10] TO BT SC BR BR BR BR BR BR BR
    0009            8c6cd         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [5 ] TO BT SC BT SA
    0010         8cbc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] TO BT BT SC BT BR BT SA
    0011             4ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BT BT AB
    0012          8cc6ccd         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0013           8cbc6d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] TO SC BT BR BT SA
    0014               4d         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO AB
    0015           86cbcd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
    0016        8cbbc6ccd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO BT BT SC BT BR BR BT SA
    0017           8b6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BT SC BR SA
    0018           8c6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO BT BR SC BT SA
    0019        8cbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT SC BR BR BR BR BT SA
    .                             100000    100000         0.00/7 =  0.00  (pval:1.000 prob:0.000)  
    .                pflags_ana  3:tboolean-box   -3:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1880     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [3 ] TO|BT|SA
    0001             1480      6312      6312             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] TO|BR|SA
    0002             1c80      5795      5795             0.00        1.000 +- 0.013        1.000 +- 0.013  [4 ] TO|BT|BR|SA
    0003             18a0        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [4 ] TO|BT|SA|SC
    0004             10a0        29        29             0.00        1.000 +- 0.186        1.000 +- 0.186  [3 ] TO|SA|SC
    0005             1808        19        19             0.00        1.000 +- 0.229        1.000 +- 0.229  [3 ] TO|BT|AB
    0006             1ca0        18        18             0.00        1.000 +- 0.236        1.000 +- 0.236  [5 ] TO|BT|BR|SA|SC
    0007             1c20        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [4 ] TO|BT|BR|SC
    0008             1008         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] TO|AB
    0009             1c08         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO|BT|BR|AB
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    .                seqmat_ana  3:tboolean-box   -3:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    0000             1232     87777     87777             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] Vm F2 Vm Rk
    0001              122      6341      6341             0.00        1.000 +- 0.013        1.000 +- 0.013  [3 ] Vm Vm Rk
    0002            12332      5427      5427             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] Vm F2 F2 Vm Rk
    0003           123332       350       350             0.00        1.000 +- 0.053        1.000 +- 0.053  [6 ] Vm F2 F2 F2 Vm Rk
    0004          1233332        28        28             0.00        1.000 +- 0.189        1.000 +- 0.189  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0005            12232        24        24             0.00        1.000 +- 0.204        1.000 +- 0.204  [5 ] Vm F2 Vm Vm Rk
    0006              332        16        16             0.00        1.000 +- 0.250        1.000 +- 0.250  [3 ] Vm F2 F2
    0007       3333333332        10        10             0.00        1.000 +- 0.316        1.000 +- 0.316  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0008          1232232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] Vm F2 Vm Vm F2 Vm Rk
    0009               22         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [2 ] Vm Vm
    0010             2232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] Vm F2 Vm Vm
    0011           123322         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [6 ] Vm Vm F2 F2 Vm Rk
    0012         12332232         3         3             0.00        1.000 +- 0.577        1.000 +- 0.577  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0013           122332         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] Vm F2 F2 Vm Vm Rk
    0014        123332232         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Vm F2 Vm Vm F2 F2 F2 Vm Rk
    0015        123333332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] Vm F2 F2 F2 F2 F2 F2 Vm Rk
    0016           123222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm F2 Vm Rk
    0017            12322         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [5 ] Vm Vm F2 Vm Rk
    0018           122232         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm F2 Vm Vm Vm Rk
    0019             3332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] Vm F2 F2 F2
    .                             100000    100000         0.00/6 =  0.00  (pval:1.000 prob:0.000)  
    ab.a.metadata:                 /tmp/blyth/opticks/evt/tboolean-box/torch/3 ca960875c829b9b5d2462c265fdb7510 c73dd7e7dad8c7e239794d2f2eda381c  100000    -1.0000 COMPUTE_MODE 
    ab.a.metadata.csgmeta0:{u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.013763847773674343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1404432/     12: 0.000  mx/mn/av 0.01376/     0/1.176e-07  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   75744/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420  108400/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    8376/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     348/      1: 0.003  mx/mn/av 0.01376/     0/3.955e-05  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :        26     728/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :        24     480/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     192/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     360/      1: 0.003  mx/mn/av 0.01376/     0/3.823e-05  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7     140/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      96/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      48/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      84/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        3  :         3      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      48/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      40/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      20/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      16/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      20/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1053324/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   56808/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420   81300/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    6282/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     261/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :        26     546/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :        24     360/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     144/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     270/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7     105/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      72/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      63/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        3  :         3      54/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      54/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      27/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      30/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      12/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      15/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:0.000457763671875 maxdv:[5.960464477539063e-08, 0.0, 5.960464477539063e-08, 5.960464477539063e-08, 0.0002593994140625, 5.960464477539063e-08, 0.000156402587890625, 7.62939453125e-06, 0.00020599365234375, 0.000396728515625, 0.000457763671875, 2.384185791015625e-07, 6.103515625e-05, 6.103515625e-05, 0.0, 9.918212890625e-05, 0.0001373291015625, 4.57763671875e-05, 9.5367431640625e-07, 9.1552734375e-05, 6.103515625e-05, 4.1961669921875e-05, 0.00017249584197998047, 0.0001373291015625, 0.0001220703125, 0.00019073486328125, 2.384185791015625e-07, 7.62939453125e-06, 0.0001220703125] 
     0000            :                    TO BT BT SA :   87777    87777  :     87777 1053324/      0: 0.000  mx/mn/av 5.96e-08/     0/4.967e-09  eps:0.0002    
     0001            :                       TO BR SA :    6312     6312  :      6312   75744/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                 TO BT BR BT SA :    5420     5420  :      5420   65040/      0: 0.000  mx/mn/av 5.96e-08/     0/4.967e-09  eps:0.0002    
     0003            :              TO BT BR BR BT SA :     349      349  :       349    4188/      0: 0.000  mx/mn/av 5.96e-08/     0/4.967e-09  eps:0.0002    
     0004            :                       TO SC SA :      29       29  :        29     348/      1: 0.003  mx/mn/av 0.0002594/     0/6.101e-06  eps:0.0002    
     0005            :           TO BT BR BR BR BT SA :      26       26  :        26     312/      0: 0.000  mx/mn/av 5.96e-08/     0/4.967e-09  eps:0.0002    
     0006            :                 TO BT BT SC SA :      24       24  :        24     288/      0: 0.000  mx/mn/av 0.0001564/     0/4.389e-06  eps:0.0002    
     0007            :                       TO BT AB :      16       16  :        16     192/      0: 0.000  mx/mn/av 7.629e-06/     0/2.39e-07  eps:0.0002    
     0008            :  TO BT SC BR BR BR BR BR BR BR :       9        9  :         9     108/      1: 0.009  mx/mn/av 0.000206/     0/1.734e-05  eps:0.0002    
     0009            :                 TO BT SC BT SA :       7        7  :         7      84/      3: 0.036  mx/mn/av 0.0003967/     0/1.953e-05  eps:0.0002    
     0010            :        TO BT BT SC BT BR BT SA :       3        3  :         3      36/      2: 0.056  mx/mn/av 0.0004578/     0/3.15e-05  eps:0.0002    
     0011            :                    TO BT BT AB :       3        3  :         3      36/      0: 0.000  mx/mn/av 2.384e-07/     0/1.821e-08  eps:0.0002    
     0012            :           TO BT BT SC BT BT SA :       3        3  :         3      36/      0: 0.000  mx/mn/av 6.104e-05/     0/4.162e-06  eps:0.0002    
     0013            :              TO SC BT BR BT SA :       3        3  :         3      36/      0: 0.000  mx/mn/av 6.104e-05/     0/6.067e-06  eps:0.0002    
     0014            :                          TO AB :       3        3  :         3      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0015            :              TO BT BR BT SC SA :       2        2  :         2      24/      0: 0.000  mx/mn/av 9.918e-05/     0/7.98e-06  eps:0.0002    
     0016            :     TO BT BT SC BT BR BR BT SA :       2        2  :         2      24/      0: 0.000  mx/mn/av 0.0001373/     0/1.469e-05  eps:0.0002    
     0017            :              TO BT BT SC BR SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 4.578e-05/     0/5.621e-06  eps:0.0002    
     0018            :              TO BT BR SC BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 9.537e-07/     0/1.425e-07  eps:0.0002    
     0019            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 9.155e-05/     0/9.682e-06  eps:0.0002    
     0020            :  TO BT BR SC BR BR BR BR BR BR :       1        1  :         1      12/      0: 0.000  mx/mn/av 6.104e-05/     0/5.097e-06  eps:0.0002    
     0021            :                 TO SC BT BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 4.196e-05/     0/8.611e-06  eps:0.0002    
     0022            :           TO BT BR SC BR BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 0.0001725/     0/1.968e-05  eps:0.0002    
     0023            :           TO BR SC BT BR BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 0.0001373/     0/1.149e-05  eps:0.0002    
     0024            :              TO BR SC BT BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 0.0001221/     0/1.787e-05  eps:0.0002    
     0025            :           TO SC BT BR BR BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 0.0001907/     0/2.625e-05  eps:0.0002    
     0026            :                    TO BT BR AB :       1        1  :         1      12/      0: 0.000  mx/mn/av 2.384e-07/     0/1.987e-08  eps:0.0002    
     0027            :                 TO BT BR BR AB :       1        1  :         1      12/      0: 0.000  mx/mn/av 7.629e-06/     0/6.755e-07  eps:0.0002    
     0028            :           TO BT SC BR BR BT SA :       1        1  :         1      12/      0: 0.000  mx/mn/av 0.0001221/     0/1.788e-05  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 0.000457763671875} pmxs_max_: 0.000457763671875  CUT ok.pdvmax 0.001  RC:0 
    [2018-08-09 22:10:18,010] p36319 {/Users/blyth/opticks/ana/nload.py:44} WARNING - np_load path_:$TMP/CRandomEngine_jump_photons.npy path:/tmp/blyth/opticks/CRandomEngine_jump_photons.npy DOES NOT EXIST 
    [2018-08-09 22:10:18,010] p36319 {/Users/blyth/opticks/ana/tboolean.py:85} WARNING - failed to load $TMP/CRandomEngine_jump_photons.npy 


