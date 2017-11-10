tboolean-media-okg4
=====================


::

    simon:cfg4 blyth$ tboolean-;tboolean-media --okg4






tboolean-media-p
-------------------

Observations

* seqhis : good agreement
* seqmat : good agreement, BUT material names are wrong 


::

    simon:cfg4 blyth$ t tboolean-media-p
    tboolean-media-p () 
    { 
        TESTNAME=${FUNCNAME/-p} tboolean-py- $*
    }
    simon:cfg4 blyth$ t tboolean-py-
    tboolean-py- () 
    { 
        tboolean.py --det ${TESTNAME} --tag $(tboolean-tag)
    }
    simon:cfg4 blyth$ 


::

    simon:cfg4 blyth$ tboolean-;tboolean-media --okg4
    simon:cfg4 blyth$ tboolean-;tboolean-media-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-media --tag 1
    ok.smry 1 
    [2017-11-10 12:37:07,459] p3674 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-media c2max 2.0 ipython False 
    [2017-11-10 12:37:07,459] p3674 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-10 12:37:08,213] p3674 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-10 12:37:08,215] p3674 {/Users/blyth/opticks/ana/ab.py:135} INFO - AB.init_point START
    [2017-11-10 12:37:08,215] p3674 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-media)  None 0 
    A tboolean-media/torch/  1 :  20171110-1235 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/1/fdom.npy 
    B tboolean-media/torch/ -1 :  20171110-1235 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-media/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Pyrex
    /tmp/blyth/opticks/tboolean-media--
    .                seqhis_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  TO SA
    0001     289569    290680             2.13  TO AB
    0002        290       292             0.01  TO SC SA
    0003         82        98             1.42  TO SC AB
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    .                pflags_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  TO|SA
    0001     289569    290680             2.13  TO|AB
    0002        290       292             0.01  TO|SA|SC
    0003         82        98             1.42  TO|SC|AB
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    .                seqmat_ana  1:tboolean-media   -1:tboolean-media        c2        ab        ba 
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
    0000     310059    308930             2.06  LS Gd
    0001     289569    290680             2.13  LS LS
    0002        290       292             0.01  LS LS Gd
    0003         82        98             1.42  LS LS LS
    .                             600000    600000         5.62/3 =  1.87  (pval:0.132 prob:0.868)  
                   /tmp/blyth/opticks/evt/tboolean-media/torch/1 2e356218300096067b2b665b354d6469 414c405507776baab2b9b0f30ae1e582  600000    -1.0000 INTEROP_MODE 
    {u'nx': u'20', u'emitconfig': u'photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'emit': -1, u'poly': u'MC'}
    [2017-11-10 12:37:08,218] p3674 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:cfg4 blyth$ 





