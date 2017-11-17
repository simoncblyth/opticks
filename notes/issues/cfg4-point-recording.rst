cfg4-point-recording CRecorder::posttrackWritePoints experiment
==================================================================

Observations
---------------

* tis much simpler
* perfect seqhis, see cfg4-bouncemax-not-working.rst
* seqmat totally messed up (no matswap)


::

     77 void CRecorder::posttrack() // invoked from CTrackingAction::PostUserTrackingAction
     78 {
     79     assert(!m_live);
     80 
     81     if(m_ctx._dbgrec) LOG(info) << "CRecorder::posttrack" ;
     82 
     83     //posttrackWriteSteps();
     84     posttrackWritePoints();  // experimental alt 
     85 
     86     if(m_dbg) m_dbg->posttrack();
     87 }




::

    tboolean-truncate-p


    [2017-11-17 18:16:56,877] p65154 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-1816 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy 
    B tboolean-truncate/torch/ -1 :  20171117-1816 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy 
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
    .                             100000    100000    199914.00/9 = 22212.67  (pval:0.000 prob:1.000)  
    0000       2222222222     99968         0         99968.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001       1111111112         0     99633         99633.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Rk Rk Rk
    0002       2111111112         0        53            53.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Rk Rk Vm
    0003       1111111212         0        42            42.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Vm Rk Rk Rk Rk Rk Rk Rk
    0004       1112111112         0        42            42.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Vm Rk Rk Rk
    0005       1111121112         0        42            42.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Vm Rk Rk Rk Rk Rk
    0006       1111211112         0        36            36.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Vm Rk Rk Rk Rk
    0007       1111111122         0        36            36.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Vm Rk Rk Rk Rk Rk Rk Rk Rk
    0008       1211111112         0        31            31.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Rk Vm Rk
    0009       1121111112         0        31            31.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Vm Rk Rk
    0010       1111112112         0        30             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Vm Rk Rk Rk Rk Rk Rk
    0011         22222222         9         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0012              212         0         6             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Vm Rk Vm
    0013          2222222         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0014            22222         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm Vm Vm Vm Vm
    0015          2111112         0         4             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Rk Rk Rk Rk Rk Vm
    0016              222         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Vm Vm Vm
    0017         21111112         0         4             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm Rk Rk Rk Rk Rk Rk Vm
    0018               22         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] Vm Vm
    0019             2112         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm Rk Rk Vm
    .                             100000    100000    199914.00/9 = 22212.67  (pval:0.000 prob:1.000)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'20', u'emit': -1}
    [2017-11-17 18:16:56,883] p65154 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:issues blyth$ 

