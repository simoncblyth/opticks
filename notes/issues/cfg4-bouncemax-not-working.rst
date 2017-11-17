cfg4-lacks-bouncemax-equivalent
=================================


ISSUE
-------

Whilst trying to make the "TO BR SA"  1st material wrong issue worse (effect a larger fraction of photons for easier debug)
I made the glass block an internal perfectSpecularSurface and emit inwards on 1 sheet if that box. 

* Opticks bouncemax prevents this going on forever

* BUT: there is no bouncemax equivalent with CFG4 ... so it proceeds to bounce, occuping 
  all machine memory and dies ! 

  * actually there is : but only in the live branch, not the default Canned approach 



APPROACH
-----------

* added step limit in CRec::add, succeeds to avoid memory death 
* BUT : seqmat/seqhis are all zero 


::

    tboolean-;tboolean-truncate --okg4 --dbgzero 



FIXED REJIG BREAKAGE
----------------------


Although fixed the seqmat/seqhis zeros getting not-done fallout in CRecorder::posttrackWriteSteps.

* All 28 of the non-truncated are falling out of the crec loop, with done=false.
  (fixed : was trivial "bool done =" vs "done =" bug in CPhoton)


::

    [2017-11-17 15:41:59,743] p54112 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-1539 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy 
    B tboolean-truncate/torch/ -1 :  20171117-1539 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    0000       aaaaaaaaad     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       6aaaaaaaad        41        49             0.71        0.837 +- 0.131        1.195 +- 0.171  [10] TO SR SR SR SR SR SR SR SR SC
    0003       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0004       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0005       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0012            4aaad         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] TO SR SR SR AB
    0013          4aaaaad         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] TO SR SR SR SR SR AB
    0014       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0015               4d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO AB
    0016        4aaaaaaad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO SR SR AB
    0018           4aaaad         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  



* thats the horrible logic... to handle bizarre BR/StepToSmall etc..


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    2017-11-17 15:29:46.728 INFO  [5445216] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-17 15:29:47.208 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 1 photon CPhoton slot_constrained 3 seqhis                 4aad seqmat                 2222 is_flag_done Y is_done Y
    2017-11-17 15:29:47.258 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 2 photon CPhoton slot_constrained 1 seqhis                   4d seqmat                   22 is_flag_done Y is_done Y
    2017-11-17 15:29:47.426 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 3 photon CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-17 15:29:47.603 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 4 photon CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-17 15:29:47.671 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 5 photon CPhoton slot_constrained 1 seqhis                   4d seqmat                   22 is_flag_done Y is_done Y
    2017-11-17 15:29:47.789 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 6 photon CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-17 15:29:48.896 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 7 photon CPhoton slot_constrained 8 seqhis            4aaaaaaad seqmat            222222222 is_flag_done Y is_done Y
    2017-11-17 15:29:49.354 FATAL [5445216] [CRecorder::posttrackWriteSteps@328] posttrackWriteSteps  not-done 8 photon CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y



REJIG BREAKAGE
-----------------


::

    [2017-11-17 14:14:02,825] p49262 {/Users/blyth/opticks/ana/seq.py:160} WARNING - SeqType.code check [?0?] bad 1 
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-1413 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy 
    B tboolean-truncate/torch/ -1 :  20171117-1413 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.11/9 =  0.46  (pval:0.904 prob:0.096)  
    0000       aaaaaaaaad     99603     99637             0.01        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0003       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0004       6aaaaaaaad        41        41             0.00        1.000 +- 0.156        1.000 +- 0.156  [10] TO SR SR SR SR SR SR SR SR SC
    0005       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0010                0         0       *28*             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] ?0?
    0011         4aaaaaad         9         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO SR SR SR SR SR SR AB
    0012            4aaad         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO SR SR SR AB
    0013          4aaaaad         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SR SR SR SR SR AB
    0014       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0015              4ad         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SR AB
    0016               4d         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO AB
    0017        4aaaaaaad         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO SR SR SR SR SR SR SR AB
    0018             4aad         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO SR SR AB
    0019           4aaaad         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.11/9 =  0.46  (pval:0.904 prob:0.096)  

    /// getting 28  G4 seqhis zeros for the non-truncated


    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000        22.27/2 = 11.13  (pval:0.000 prob:1.000)  
    0000             1200     99603     99588             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SR
    0001             1220       361       380             0.49        0.950 +- 0.050        1.053 +- 0.054  [3 ] TO|SR|SC
    0002             1208        32         4            21.78        8.000 +- 1.414        0.125 +- 0.062  [3 ] TO|SR|AB
    0003                0         0        28             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ]
    0004             1008         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|AB
    .                             100000    100000        22.27/2 = 11.13  (pval:0.000 prob:1.000)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000       2222222222     99968     99972             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001                0         0        28             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] ?0?
    0002         22222222         9         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0003          2222222         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0004            22222         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm Vm Vm Vm Vm
    0005              222         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Vm Vm Vm
    0006               22         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] Vm Vm
    0007        222222222         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0008             2222         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm Vm Vm Vm
    0009           222222         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm Vm Vm Vm Vm Vm
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 





FIXED EXPENSIVELY : by ~doubling step_limit to cope with G4 StepTooSmall turnarounds 
------------------------------------------------------------------------------------------

::

    .unsigned CG4Ctx::step_limit() const 
     {
    -    return 1 + ( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
    +    return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
     }
     
 

* G4 burns thru steps for BR, with loada "StepTooSmall" turnarounds at the reflections
  so in order to reach normal truncation have to save more steps 

* this expensive fix to CRecorder::posttrackWriteSteps suggests that should 
  be step-by-step collecting G4StepPoint with the skips done upfront rather than collecting G4Step
  only to "StepTooSmall" skip them later 




::


    tboolean-truncate--(){ cat << EOP 
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="$TMP/$FUNCNAME")

    emitconfig = "photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1" 

    CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig=emitconfig  )

    box = CSG("box", param=[0,0,0,200], emit=-1,  boundary="Rock//perfectSpecularSurface/Vacuum" )

    CSG.Serialize([box], args.csgpath )
    EOP
    }


    [2017-11-17 11:15:54,433] p39265 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-1115 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy 
    B tboolean-truncate/torch/ -1 :  20171117-1115 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    0000       aaaaaaaaad     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       6aaaaaaaad        41        49             0.71        0.837 +- 0.131        1.195 +- 0.171  [10] TO SR SR SR SR SR SR SR SR SC
    0003       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0004       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0005       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0012            4aaad         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] TO SR SR SR AB
    0013          4aaaaad         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] TO SR SR SR SR SR AB
    0014       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0015               4d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO AB
    0016        4aaaaaaad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO SR SR AB
    0018           4aaaad         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    0000             1200     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SR
    0001             1220       361       339             0.69        1.065 +- 0.056        0.939 +- 0.051  [3 ] TO|SR|SC
    0002             1208        32        25             0.86        1.280 +- 0.226        0.781 +- 0.156  [3 ] TO|SR|AB
    0003             1008         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO|AB
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000       2222222222     99968     99976             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001         22222222         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0002              222         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] Vm Vm Vm
    0003          2222222         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0004            22222         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] Vm Vm Vm Vm Vm
    0005               22         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] Vm Vm
    0006        222222222         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0007             2222         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] Vm Vm Vm Vm
    0008           222222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm Vm Vm Vm
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'20', u'emit': -1}
    [2017-11-17 11:15:54,437] p39265 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:issues blyth$ 




tboolean-truncate-p
---------------------------

::

    [2017-11-16 21:04:05,972] p35253 {/Users/blyth/opticks/ana/seq.py:160} WARNING - SeqType.code check [?0?] bad 1 
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171116-2103 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy 
    B tboolean-truncate/torch/ -1 :  20171116-2103 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000    199936.00/10 = 19993.60  (pval:0.000 prob:1.000)  
    0000                0         0     99972         99972.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] ?0?
    0001       aaaaaaaaad     99603         0         99603.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SR SR SR SR SR
    0002       aaa6aaaaad        49         0            49.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SR SC SR SR SR
    0003       aaaaa6aaad        45         0            45.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SC SR SR SR SR SR
    0004       6aaaaaaaad        41         0            41.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SR SR SR SR SC
    0005       aaaaaa6aad        40         0            40.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39         0            39.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38         0            38.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38         0            38.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36         0            36.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SR SR SC SR SR
    0010       aaaaaaa6ad        35         0            35.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SC SR SR SR SR SR SR SR
    0011         4aaaaaad         9         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO SR SR SR SR SR SR AB
    0012            4aaad         5         7             0.00        0.714 +- 0.319        1.400 +- 0.529  [5 ] TO SR SR SR AB
    0013              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0014               4d         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [2 ] TO AB
    0015          4aaaaad         5         3             0.00        1.667 +- 0.745        0.600 +- 0.346  [7 ] TO SR SR SR SR SR AB
    0016       4aaaaaaaad         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SR SR SR SR SR SR SR SR AB
    0017           4aaaad         1         3             0.00        0.333 +- 0.333        3.000 +- 1.732  [6 ] TO SR SR SR SR AB
    0018             4aad         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [4 ] TO SR SR AB
    0019        4aaaaaaad         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO SR SR SR SR SR SR SR AB
    .                             100000    100000    199936.00/10 = 19993.60  (pval:0.000 prob:1.000)  
    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000    199937.85/3 = 66645.95  (pval:0.000 prob:1.000)  
    0000                0         0     99972         99972.00        0.000 +- 0.000        0.000 +- 0.000  [1 ]
    0001             1200     99603         0         99603.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|SR
    0002             1220       361         0           361.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SR|SC
    0003             1208        32        22             1.85        1.455 +- 0.257        0.688 +- 0.147  [3 ] TO|SR|AB
    0004             1008         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [2 ] TO|AB
    .                             100000    100000    199937.85/3 = 66645.95  (pval:0.000 prob:1.000)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000    199940.00/1 = 199940.00  (pval:0.000 prob:1.000)  
    0000                0         0     99972         99972.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] ?0?
    0001       2222222222     99968         0         99968.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0002         22222222         9         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0003            22222         5         7             0.00        0.714 +- 0.319        1.400 +- 0.529  [5 ] Vm Vm Vm Vm Vm
    0004              222         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] Vm Vm Vm
    0005               22         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [2 ] Vm Vm
    0006          2222222         5         3             0.00        1.667 +- 0.745        0.600 +- 0.346  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0007           222222         1         3             0.00        0.333 +- 0.333        3.000 +- 1.732  [6 ] Vm Vm Vm Vm Vm Vm
    0008             2222         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [4 ] Vm Vm Vm Vm
    0009        222222222         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] Vm Vm Vm Vm Vm Vm Vm Vm Vm
    .                             100000    100000    199940.00/1 = 199940.00  (pval:0.000 prob:1.000)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'








TESTS
--------

::

    tboolean-;tboolean-box --okg4 --steppingdbg -D


    (lldb) b CRecorder::Record(G4Step const*, int, int, bool, bool, DsG4OpBoundaryProcessStatus, CStage::CStage_t) 


    (lldb) b CRecorder::CannedRecordStep()

    (lldb) p m_crec->getNumStps()
    (unsigned int) $8 = 2790




GEOM
------

::

    tboolean-box--(){ cat << EOP 
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="$TMP/$FUNCNAME")

    emitconfig = "photons=1000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1" 

    CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig=emitconfig  )

    container = CSG("box", emit=0, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing

    box = CSG("box3", param=[300,300,200,0], emit=-1,  boundary="Vacuum//perfectSpecularSurface/GlassSchottF2" )

    CSG.Serialize([container, box], args.csgpath )
    EOP
    }



REVIEW
----------




CFG4::

    210 
    211 void CSteppingAction::UserSteppingAction(const G4Step* step)
    212 {
    213     int step_id = CTrack::StepId(m_track);
    214     bool done = setStep(step, step_id);
    215 
    216     if(done)
    217     {
    218         G4Track* track = step->GetTrack();    // m_track is const qualified
    219         track->SetTrackStatus(fStopAndKill);
    220         // stops tracking when reach truncation as well as absorption
    221     }
    222 }
    223 


    230 bool CSteppingAction::setStep(const G4Step* step, int step_id)
    231 {
    232     bool done = false ;
    233 
    234     m_step = step ;
    235     m_step_id = step_id ;
    236 
    237     if(m_step_id == 0)
    238     {
    239         const G4StepPoint* pre = m_step->GetPreStepPoint() ;
    240         m_step_origin = pre->GetPosition();
    241     }
    242 
    243 
    244     m_track_step_count += 1 ;
    245     m_step_total += 1 ;
    246 
    247     G4TrackStatus track_status = m_track->GetTrackStatus();
    248 
    249     LOG(trace) << "CSteppingAction::setStep"
    250               << " step_total " << m_step_total
    251               << " event_id " << m_event_id
    252               << " track_id " << m_track_id
    253               << " track_step_count " << m_track_step_count
    254               << " step_id " << m_step_id
    255               << " trackStatus " << CTrack::TrackStatusString(track_status)
    256               ;
    257 
    258     if(m_optical)
    259     {
    260         done = collectPhotonStep();
    261     }
    262     else
    263     {
    264         m_steprec->collectStep(step, step_id);
    265    
    266         if(track_status == fStopAndKill)
    267         {
    268             done = true ;
    269             m_steprec->storeStepsCollected(m_event_id, m_track_id, m_pdg_encoding);
    270             m_steprec_store_count = m_steprec->getStoreCount();
    271         }
    272     }
    273 
    274    if(m_step_total % 10000 == 0)
    275        LOG(debug) << "CSA (totals%10k)"
    276                  << " track_total " <<  m_track_total
    277                  << " step_total " <<  m_step_total
    278                  ;
    279 
    280     return done ;



    284 bool CSteppingAction::collectPhotonStep()
    285 {
    286     bool done = false ;
    287 
    288 
    289     CStage::CStage_t stage = CStage::UNKNOWN ;
    290 
    291     if( !m_reemtrack )     // primary photon, ie not downstream from reemission 
    292     {
    293         stage = m_primarystep_count == 0  ? CStage::START : CStage::COLLECT ;
    294         m_primarystep_count++ ;
    295     }
    296     else
    297     {
    298         stage = m_rejoin_count == 0  ? CStage::REJOIN : CStage::RECOLL ;
    299         m_rejoin_count++ ;
    300         // rejoin count is zeroed in setPhotonId, so each remission generation trk will result in REJOIN 
    301     }
    302 
    303 
    304     // TODO: avoid need for these
    305     m_recorder->setPhotonId(m_photon_id);
    306     m_recorder->setEventId(m_event_id);
    307 
    308     int record_max = m_recorder->getRecordMax() ;
    309     bool recording = m_record_id < record_max ||  m_dynamic ;
    310 
    311     if(recording)
    312     {
    313 #ifdef USE_CUSTOM_BOUNDARY
    314         DsG4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
    315 #else
    316         G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
    317 #endif
    318         done = m_recorder->Record(m_step, m_step_id, m_record_id, m_debug, m_other, boundary_status, stage);
    319 
    320     }
    321     // hmm perhaps the recording restriction is why bouncemax doesnt kick in ? for the infini-bouncers
    322     return done ;
    323 }







bouncemax::

    simon:cfg4 blyth$ opticks-find bouncemax 
    ./ok/ok.bash:    ggv --jpmt --modulo 1000 --bouncemax 0
    ./ok/ok.bash:    ggv --jpmt --modulo 1000 --bouncemax 0
    ./ok/ok.bash:    ggv --make --jpmt --modulo 100 --override 5182 --debugidx 5181 --bouncemax 0 
    ./optixrap/oxrap.bash:ISSUE: restricting bouncemax prevents recsel selection operation
    ./tests/tboolean.bash:  the bouncemax prevents this going on forever, but there is 
    ./tests/tconcentric.bash:    tconcentric-t --bouncemax 15 --recordmax 16 --groupvel --finebndtex $* 
    ./optickscore/OpticksCfg.cc:       m_bouncemax(9),     
    ./optickscore/OpticksCfg.cc:   char bouncemax[128];
    ./optickscore/OpticksCfg.cc:   snprintf(bouncemax,128, 
    ./optickscore/OpticksCfg.cc:"Default %d ", m_bouncemax);
    ./optickscore/OpticksCfg.cc:       ("bouncemax,b",  boost::program_options::value<int>(&m_bouncemax), bouncemax );
    ./optickscore/OpticksCfg.cc:   // keeping bouncemax one less than recordmax is advantageous 
    ./optickscore/OpticksCfg.cc:    return m_bouncemax ; 
    ./optickscore/OpticksCfg.hh:     int         m_bouncemax ; 
    ./ana/debug/genstep_sequence_material_mismatch.py:    ggv --torchconfig "zenith_azimuth:0,0.31,0,1" --bouncemax 1
    simon:opticks blyth$ 

    simon:opticks blyth$ opticks-find getBounceMax
    ./cfg4/CRecorder.cc:    m_bounce_max = m_evt->getBounceMax();
    ./optickscore/Opticks.cc:    unsigned int bounce_max = getBounceMax() ;
    ./optickscore/Opticks.cc:unsigned Opticks::getBounceMax() {   return m_cfg->getBounceMax(); }
    ./optickscore/OpticksCfg.cc:int OpticksCfg<Listener>::getBounceMax()
    ./optickscore/OpticksEvent.cc:unsigned int OpticksEvent::getBounceMax()
    ./optixrap/OPropagator.cc:    m_context[ "bounce_max" ]->setUint( m_ok->getBounceMax() );
    ./optickscore/Opticks.hh:       unsigned getBounceMax();
    ./optickscore/OpticksCfg.hh:     int          getBounceMax(); 
    ./optickscore/OpticksEvent.hh:       unsigned int getBounceMax();
    simon:opticks blyth$ 


::

     354 void CRecorder::initEvent(OpticksEvent* evt)
     355 {
     356     setEvent(evt);
     357 
     358     m_c4.u = 0u ;
     359 
     360     m_record_max = m_evt->getNumPhotons();   // from the genstep summation
     361     m_bounce_max = m_evt->getBounceMax();
     362 


Huh, looks like there is bounce truncate ?::

     836 #ifdef USE_CUSTOM_BOUNDARY
     837 bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label)
     838 #else
     839 bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
     840 #endif
     841 {
     842     // see notes/issues/geant4_opticks_integration/tconcentric_pflags_mismatch_from_truncation_handling.rst
     843     //
     844     // NB this is used by both the live and non-live "canned" modes of recording 
     845     //
     846     // Formerly at truncation, rerunning this overwrote "the top slot" 
     847     // of seqhis,seqmat bitfields (which are persisted in photon buffer)
     848     // and the record buffer. 
     849     // As that is different from Opticks behaviour for the record buffer
     850     // where truncation is truncation, a HARD_TRUNCATION has been adopted.
     ...
     933 
     934     RecordStepPoint(slot, point, flag, material, label);
     935 
     936     double time = point->GetGlobalTime();
     937 
     938 
     939     if(m_debug || m_other) Collect(point, flag, material, boundary_status, m_mskhis, m_seqhis, m_seqmat, time);
     940 
     941     m_slot += 1 ;    // m_slot is incremented regardless of truncation, only local *slot* is constrained to recording range
     942 
     943     m_bounce_truncate = m_slot > m_bounce_max  ;
     944     if(m_bounce_truncate) m_step_action |= BOUNCE_TRUNCATE ;
     945 
     946 
     947     bool done = m_bounce_truncate || m_record_truncate || absorb || miss ;
     948 
     949     if(done && m_dynamic)
     950     {
     951         m_records->add(m_dynamic_records);
     952     }
     953 
     954     return done ;   
     955 }

