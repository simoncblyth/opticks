FIXED RNG_seq_off_by_one : maligned six
==========================================


TODO
------

* try to simplify the kludge, eg by removing inhibitions and adjusting peek offset 

  * have tried, seems need the triple-whammy approach 
  * can definitely improve code clarity and switch the options ON  default

* un-conflate zero-steps and StepTooSmall

* current dbgkludgeflatzero uses hard coded peek(-3), 
  need a better way : as other kludges will effect the appropriate peek back


ASIS 
-----


DONE
------

* fix the python debug comparison to be aware of the kludge


Full running 
----------------

* full report :doc:`tboolean_box_perfect_alignment`


FIXED Issue : Dirty Half Dozen out of alignment
--------------------------------------------------

::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


::

    tboolean-;tboolean-box --okg4 --align --mask 0  --pindex 0 --pindexlog -DD   

    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 9041  --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 14510 --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 49786 --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog -DD   


    tboolean-;tboolean-box --okg4 --align --mask 1230,9041,14510,49786,69653,77962 -DD

    ucf.py 9041
    bouncelog.py 9041


Following the devious triple kludge
--------------------------------------

Devious triple whammy kludge --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero manages to persuade G4 to match Opticks histories of the maligned six.

::

    tboolean-;tboolean-box --okg4 --align --mask 1230,9041,14510,49786,69653,77962 --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero

    In [1]: ab.maligned    ## this is just on the 6, not a full check 
    Out[1]: array([], dtype=int64)

    In [2]: ab.dumpline(range(0,6))
          0      0 :   :                               TO BR SC BT BR BT SA                               TO BR SC BT BR BT SA 
          1      1 :   :                         TO BT SC BR BR BR BR BT SA                         TO BT SC BR BR BR BR BT SA 
          2      2 :   :                               TO SC BT BR BR BT SA                               TO SC BT BR BR BT SA 
          3      3 :   :                         TO BT BT SC BT BR BR BT SA                         TO BT BT SC BT BR BR BT SA 
          4      4 :   :                               TO BT SC BR BR BT SA                               TO BT SC BR BR BT SA 
          5      5 :   :                               TO BT BR SC BR BT SA                               TO BT BR SC BR BT SA 


::

    2017-12-16 18:37:15.749 INFO  [1093268] [OpticksAna::run@66] OpticksAna::run anakey tboolean enabled Y
    args: /Users/blyth/opticks/ana/tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch
    [2017-12-16 18:37:16,085] p15240 {/Users/blyth/opticks/ana/tboolean.py:62} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2017-12-16 18:37:16,085] p15240 {/Users/blyth/opticks/ana/ab.py:110} INFO - ab START
    ab.a.metadata:                 /tmp/blyth/opticks/evt/tboolean-box/torch/1 5bfbabf976e0dd1acd15cb74901a868e 538275366882781e5c03160c15cd9f08       6    -1.0000 INTEROP_MODE 
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171216-1837 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171216-1837 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                                  6         6         0.00/-1 =  0.00  (pval:nan prob:nan)  
    0000          8cb6bcd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] TO BT BR SC BR BT SA
    0001          8cbc6bd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] TO BR SC BT BR BT SA
    0002        8cbbc6ccd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT BT SC BT BR BR BT SA
    0003          8cbbc6d         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] TO SC BT BR BR BT SA
    0004        8cbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] TO BT SC BR BR BR BR BT SA
    0005          8cbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] TO BT SC BR BR BT SA
    .                                  6         6         0.00/-1 =  0.00  (pval:nan prob:nan)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                                  6         6         0.00/-1 =  0.00  (pval:nan prob:nan)  
    0000             1ca0         6         6             0.00        1.000 +- 0.408        1.000 +- 0.408  [5 ] TO|BT|BR|SA|SC
    .                                  6         6         0.00/-1 =  0.00  (pval:nan prob:nan)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                                  6         6         0.00/-1 =  0.00  (pval:nan prob:nan)  
    0000          1233332         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0001          1233222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] Vm Vm Vm F2 F2 Vm Rk
    0002        123332232         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] Vm F2 Vm Vm F2 F2 F2 Vm Rk
    0003          1233322         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [7 ] Vm Vm F2 F2 F2 Vm Rk
    0004        123333332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [9 ] Vm F2 F2 F2 F2 F2 F2 Vm Rk
    .                                  6         6         0.00/-1 =  0.00  (pval:nan prob:nan)  
    ab.a.metadata:                 /tmp/blyth/opticks/evt/tboolean-box/torch/1 5bfbabf976e0dd1acd15cb74901a868e 538275366882781e5c03160c15cd9f08       6    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0:{u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :           TO BT BR SC BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :           TO BR SC BT BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :     TO BT BT SC BT BR BR BT SA :       1        1  :         1      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :           TO SC BT BR BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      36/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0005            :           TO BT SC BR BR BT SA :       1        1  :         1      28/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :           TO BT BR SC BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :           TO BR SC BT BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :     TO BT BT SC BT BR BR BT SA :       1        1  :         1      27/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :           TO SC BT BR BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      27/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0005            :           TO BT SC BR BR BT SA :       1        1  :         1      21/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:0.000190734863281 maxdv:[0.00016832351684570312, 0.0001373291015625, 6.103515625e-05, 0.00019073486328125, 0.0001220703125, 0.00018310546875] 
     0000            :           TO BT BR SC BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001683/     0/1.737e-05  eps:0.0002    
     0001            :           TO BR SC BT BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001373/     0/8.614e-06  eps:0.0002    
     0002            :     TO BT BT SC BT BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 6.104e-05/     0/7.655e-06  eps:0.0002    
     0003            :           TO SC BT BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001907/     0/1.969e-05  eps:0.0002    
     0004            :     TO BT SC BR BR BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001221/     0/1.114e-05  eps:0.0002    
     0005            :           TO BT SC BR BR BT SA :       1        1  :         1      16/      0: 0.000  mx/mn/av 0.0001831/     0/1.821e-05  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.0} rmxs_max_: 0.0  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 0.00019073486328125} pmxs_max_: 0.000190734863281  CUT ok.pdvmax 0.001  RC:0 
    [2017-12-16 18:37:16,210] p15240 {/Users/blyth/opticks/ana/tboolean.py:70} INFO - early exit as non-interactive
    2017-12-16 18:37:16.241 INFO  [1093268] [SSys::run@50] tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch   rc_raw : 0 rc : 0
    2017-12-16 18:37:16.242 INFO  [1093268] [OpticksAna::run@79] OpticksAna::run anakey tboolean cmdline tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch   rc 0 rcmsg -



Full unmasked run into tag 2 : To find some jump record_id
--------------------------------------------------------------------------------------------

* Obtain indices of all photons with jump backs and study them.

::

    tboolean-;TBOOLEAN_TAG=2 tboolean-box --okg4 --align 
    tboolean-;TBOOLEAN_TAG=2 tboolean-box-ip



OpBoundary value actually arbitrary
--------------------------------------

Notice that the value returned for OpBoundary does not matter, 
as no interaction length is actually used by either Opticks or Geant4, 
it is just required to keep the sequences aligned.

Thus returning zero, which is never given by curand_uniform is a good choice.

::

    __device__ float 
    curand_uniform (curandState_t *state)
    This function returns a sequence of pseudorandom floats uniformly distributed
    between 0.0 and 1.0. It may return from 0.0 to 1.0, where 1.0 is included and
    0.0 is excluded.

    Read more at: http://docs.nvidia.com/cuda/curand/index.html



fixing the python debug comparison
------------------------------------

Change correspondence between seqs to use m_cursor
rather than m_current_record_flat_count as 
m_cursor is untainted by kludges.

The location is still discrepant where the kludge is applied.


g4lldb.py::

    216     def flat(self, ploc, frame, bp_loc, sess):
    217         self.v = frame.FindVariable("this")
    218                
    219         ug4 = self.ev("m_flat")
    220         lg4 = self.ev("m_location")
    221         cur = self.ev("m_cursor")
    222         crf = self.ev("m_current_record_flat_count")
    223         csf = self.ev("m_current_step_flat_count")
    224         cix = self.ev("m_curand_index")
    225     
    226         idx = cur
    227         # correspondence between sequences 
    228         #    crf: gets offset by the kludge
    229         #    cur: is always the real flat cursor index
    230 
    231         assert type(crf) is int
    232         assert type(csf) is int
    233         assert type(cix) is int and cix == self.pindex
    234     
    235         u = self.ucf[idx] if idx < self.lucf else None
    236         uok = u.fval if u is not None else -1
    237         lok = u.lab  if u is not None else "ucf-overflow"



1230 bare : ie with clears on every step and zero-step rewinds : goes off rails at TIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the below output relies on ana/g4lldb.py (unfortunately) and lldb python scripted 
breakpoints with access to C++ program context.   


::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog  -DD --noviz


    2017-12-17 14:29:34.308 INFO  [1201428] [CRandomEngine::preTrack@396] CRandomEngine::preTrack : DONE cmd "ucf.py 1230"
    CRandomEngine_cc_preTrack.[00] lucf:29 pindex:1230
    2017-12-17 14:29:34.321 ERROR [1201428] [CRandomEngine::preTrack@406] CRandomEngine::pretrack record_id:  ctx.record_id 0 use_index 1230 align_mask YES
    CRandomEngine_cc_flat.[00] cix: 1230 mrk:-- cur: 0 crf: 0 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   5.052794121e-14 ug4/ok:( 0.001117025 0.001117025 ) 
    CRandomEngine_cc_flat.[01] cix: 1230 mrk:-- cur: 1 crf: 1 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.976989766e-10 ug4/ok:( 0.502647340 0.502647340 ) 
    CRandomEngine_cc_flat.[02] cix: 1230 mrk:-- cur: 2 crf: 2 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   5.276490356e-11 ug4/ok:( 0.601504147 0.601504147 ) 
    CRandomEngine_cc_flat.[03] cix: 1230 mrk:-- cur: 3 crf: 3 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   3.701783324e-11 ug4/ok:( 0.938713491 0.938713491 ) 
    CRandomEngine_cc_postStep.[00] step_id:0 bst:FresnelReflection pri:      Undefined okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[04] cix: 1230 mrk:-- cur: 4 crf: 4 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 ug4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] cix: 1230 mrk:-- cur: 5 crf: 5 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 ug4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] cix: 1230 mrk:-- cur: 6 crf: 6 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 ug4/ok:( 0.438019574 0.438019574 ) 
    2017-12-17 14:29:34.359 INFO  [1201428] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 14:29:34.359 INFO  [1201428] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 14:29:34.359 ERROR [1201428] [CRandomEngine::postStep@323] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero NO
    CRandomEngine_cc_jump.[00] cursor_old:6 jump_:-3 jump_count:1 cursor:3 
    CRandomEngine_cc_postStep.[01] step_id:1 bst:   StepTooSmall pri:FresnelReflection okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[07] cix: 1230 mrk:-- cur: 4 crf: 7 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 ug4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[08] cix: 1230 mrk:-- cur: 5 crf: 8 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 ug4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[09] cix: 1230 mrk:-- cur: 6 crf: 9 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 ug4/ok:( 0.438019574 0.438019574 ) 
    CRandomEngine_cc_flat.[10] cix: 1230 mrk:-- cur: 7 crf:10 csf: 3 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.102905545e-10 ug4/ok:( 0.714031577 0.714031577 ) 
    CRandomEngine_cc_flat.[11] cix: 1230 mrk:-- cur: 8 crf:11 csf: 4 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.093353269e-10 ug4/ok:( 0.330403954 0.330403954 ) 
    CRandomEngine_cc_flat.[12] cix: 1230 mrk:-- cur: 9 crf:12 csf: 5 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.423827971e-10 ug4/ok:( 0.570741653 0.570741653 ) 
    CRandomEngine_cc_flat.[13] cix: 1230 mrk:-- cur:10 crf:13 csf: 6 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.903991964e-10 ug4/ok:( 0.375908673 0.375908673 ) 
    CRandomEngine_cc_flat.[14] cix: 1230 mrk:-- cur:11 crf:14 csf: 7 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.353455126e-10 ug4/ok:( 0.784978330 0.784978330 ) 
    CRandomEngine_cc_postStep.[02] step_id:2 bst:  NotAtBoundary pri:   StepTooSmall okevt_pt:TO BR [SC] BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[15] cix: 1230 mrk:-- cur:12 crf:15 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.406677163e-10 ug4/ok:( 0.892654359 0.892654359 ) 
    CRandomEngine_cc_flat.[16] cix: 1230 mrk:-- cur:13 crf:16 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.669952203e-10 ug4/ok:( 0.441063195 0.441063195 ) 
    CRandomEngine_cc_flat.[17] cix: 1230 mrk:-- cur:14 crf:17 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.626708933e-10 ug4/ok:( 0.773742437 0.773742437 ) 
    CRandomEngine_cc_flat.[18] cix: 1230 mrk:-- cur:15 crf:18 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   4.671020237e-10 ug4/ok:( 0.556839108 0.556839108 ) 
    CRandomEngine_cc_postStep.[03] step_id:3 bst:FresnelRefraction pri:  NotAtBoundary okevt_pt:TO BR SC [BT] BR BT SA                             
    --
    CRandomEngine_cc_flat.[19] cix: 1230 mrk:-- cur:16 crf:19 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:    1.88293825e-11 ug4/ok:( 0.775349319 0.775349319 ) 
    CRandomEngine_cc_flat.[20] cix: 1230 mrk:-- cur:17 crf:20 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.589111148e-10 ug4/ok:( 0.752141237 0.752141237 ) 
    CRandomEngine_cc_flat.[21] cix: 1230 mrk:-- cur:18 crf:21 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.373718527e-10 ug4/ok:( 0.412002385 0.412002385 ) 
    CRandomEngine_cc_postStep.[04] step_id:4 bst:TotalInternalReflection pri:FresnelRefraction okevt_pt:TO BR SC BT [BR] BT SA                             
    --
    off rails here ... 

    CRandomEngine_cc_flat.[22] cix: 1230 mrk:-# cur:19 crf:22 csf: 0 lg4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:   4.672088827e-10 ug4/ok:( 0.282463104 0.282463104 ) 
    CRandomEngine_cc_flat.[23] cix: 1230 mrk:-# cur:20 crf:23 csf: 1 lg4/ok: (                     OpRayleigh                     OpBoundary ) df:   1.872253463e-10 ug4/ok:( 0.432497680 0.432497680 ) 
    CRandomEngine_cc_flat.[24] cix: 1230 mrk:-# cur:21 crf:24 csf: 2 lg4/ok: (                   OpAbsorption                     OpRayleigh ) df:   4.039001356e-10 ug4/ok:( 0.907848895 0.907848895 ) 
    2017-12-17 14:29:34.443 INFO  [1201428] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 14:29:34.443 INFO  [1201428] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 14:29:34.443 ERROR [1201428] [CRandomEngine::postStep@323] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero NO
    CRandomEngine_cc_jump.[01] cursor_old:21 jump_:-3 jump_count:2 cursor:18 
    CRandomEngine_cc_postStep.[05] step_id:5 bst:   StepTooSmall pri:TotalInternalReflection okevt_pt:TO BR SC BT [BR] BT SA                             
    --
    CRandomEngine_cc_flat.[25] cix: 1230 mrk:-# cur:19 crf:25 csf: 0 lg4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:   4.672088827e-10 ug4/ok:( 0.282463104 0.282463104 ) 
    CRandomEngine_cc_flat.[26] cix: 1230 mrk:-# cur:20 crf:26 csf: 1 lg4/ok: (                     OpRayleigh                     OpBoundary ) df:   1.872253463e-10 ug4/ok:( 0.432497680 0.432497680 ) 
    CRandomEngine_cc_flat.[27] cix: 1230 mrk:-# cur:21 crf:27 csf: 2 lg4/ok: (                   OpAbsorption                     OpRayleigh ) df:   4.039001356e-10 ug4/ok:( 0.907848895 0.907848895 ) 
    CRandomEngine_cc_flat.[28] cix: 1230 mrk:-# cur:22 crf:28 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff                   OpAbsorption ) df:   7.296752091e-11 ug4/ok:( 0.912139237 0.912139237 ) 
    CRandomEngine_cc_postStep.[06] step_id:6 bst:FresnelReflection pri:   StepTooSmall okevt_pt:TO BR SC BT BR [BT] SA                             
    --
    CRandomEngine_cc_flat.[29] cix: 1230 mrk:-# cur:23 crf:29 csf: 0 lg4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:   8.567047072e-11 ug4/ok:( 0.201808557 0.201808557 ) 
    CRandomEngine_cc_flat.[30] cix: 1230 mrk:-# cur:24 crf:30 csf: 1 lg4/ok: (                     OpRayleigh                     OpBoundary ) df:   4.876709037e-10 ug4/ok:( 0.795349360 0.795349360 ) 
    CRandomEngine_cc_flat.[31] cix: 1230 mrk:-# cur:25 crf:31 csf: 2 lg4/ok: (                   OpAbsorption                     OpRayleigh ) df:   2.741393779e-10 ug4/ok:( 0.484203994 0.484203994 ) 
    2017-12-17 14:29:34.479 INFO  [1201428] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 14:29:34.479 INFO  [1201428] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 14:29:34.479 ERROR [1201428] [CRandomEngine::postStep@323] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero NO
    CRandomEngine_cc_jump.[02] cursor_old:25 jump_:-3 jump_count:3 cursor:22 
    CRandomEngine_cc_postStep.[07] step_id:7 bst:   StepTooSmall pri:FresnelReflection okevt_pt:TO BR SC BT BR [BT] SA                             
    --
    CRandomEngine_cc_flat.[32] cix: 1230 mrk:-# cur:23 crf:32 csf: 0 lg4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:   8.567047072e-11 ug4/ok:( 0.201808557 0.201808557 ) 
    CRandomEngine_cc_flat.[33] cix: 1230 mrk:-# cur:24 crf:33 csf: 1 lg4/ok: (                     OpRayleigh                     OpBoundary ) df:   4.876709037e-10 ug4/ok:( 0.795349360 0.795349360 ) 
    CRandomEngine_cc_flat.[34] cix: 1230 mrk:-# cur:25 crf:34 csf: 2 lg4/ok: (                   OpAbsorption                     OpRayleigh ) df:   2.741393779e-10 ug4/ok:( 0.484203994 0.484203994 ) 
    CRandomEngine_cc_flat.[35] cix: 1230 mrk:-# cur:26 crf:35 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff                   OpAbsorption ) df:   4.411544741e-11 ug4/ok:( 0.093548603 0.093548603 ) 
    CRandomEngine_cc_postStep.[08] step_id:8 bst:FresnelRefraction pri:   StepTooSmall okevt_pt:TO BR SC BT BR BT [SA]                             
    --
    CRandomEngine_cc_flat.[36] cix: 1230 mrk:-# cur:27 crf:36 csf: 0 lg4/ok: (                     OpBoundary OpBoundary_DiDiReflectOrTransmit ) df:    4.29260294e-10 ug4/ok:( 0.750533462 0.750533462 ) 
    CRandomEngine_cc_flat.[37] cix: 1230 mrk:-# cur:28 crf:37 csf: 1 lg4/ok: (                     OpRayleigh        OpBoundary_DoAbsorption ) df:   3.650513225e-10 ug4/ok:( 0.946246266 0.946246266 ) 
    CRandomEngine_cc_flat.[38] cix: 1230 mrk:*# cur:29 crf:38 csf: 2 lg4/ok: (                   OpAbsorption                   ucf-overflow ) df:       1.357590944 ug4/ok:( 0.357590944 -1.000000000 ) 
    CRandomEngine_cc_flat.[39] cix: 1230 mrk:*# cur:30 crf:39 csf: 3 lg4/ok: ( OpBoundary_DiDiReflectOrTransmit                   ucf-overflow ) df:       1.166174248 ug4/ok:( 0.166174248 -1.000000000 ) 
    CRandomEngine_cc_flat.[40] cix: 1230 mrk:*# cur:31 crf:40 csf: 4 lg4/ok: (        OpBoundary_DoAbsorption                   ucf-overflow ) df:       1.628916502 ug4/ok:( 0.628916502 -1.000000000 ) 
    CRandomEngine_cc_postStep.[09] step_id:9 bst:     Absorption pri:FresnelRefraction okevt_pt:TO BR SC BT BR BT SA [  ]                          
    --
    //                                                  CRandomEngine_cc_postTrack.[00] : postTrack label 
    CRandomEngine_cc_postTrack.[00] pindex:1230



1230 without --dbgskipclearzero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without the clear G4 messes up seq via (OpBoundary, OpRayleigh, OpAbsorption) yielding 
a different decision, so operating without the clear would require a jump back.


::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog  -DD --dbgnojumpzero --dbgkludgeflatzero

    2017-12-17 14:16:19.380 INFO  [1197696] [CRandomEngine::preTrack@396] CRandomEngine::preTrack : DONE cmd "ucf.py 1230"
    CRandomEngine_cc_preTrack.[00] lucf:29 pindex:1230
    2017-12-17 14:16:19.394 ERROR [1197696] [CRandomEngine::preTrack@406] CRandomEngine::pretrack record_id:  ctx.record_id 0 use_index 1230 align_mask YES
    CRandomEngine_cc_flat.[00] cix: 1230 mrk:-- cur: 0 crf: 0 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   5.052794121e-14 ug4/ok:( 0.001117025 0.001117025 ) 
    CRandomEngine_cc_flat.[01] cix: 1230 mrk:-- cur: 1 crf: 1 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.976989766e-10 ug4/ok:( 0.502647340 0.502647340 ) 
    CRandomEngine_cc_flat.[02] cix: 1230 mrk:-- cur: 2 crf: 2 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   5.276490356e-11 ug4/ok:( 0.601504147 0.601504147 ) 
    CRandomEngine_cc_flat.[03] cix: 1230 mrk:-- cur: 3 crf: 3 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   3.701783324e-11 ug4/ok:( 0.938713491 0.938713491 ) 
    CRandomEngine_cc_postStep.[00] step_id:0 bst:FresnelReflection pri:      Undefined okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[04] cix: 1230 mrk:-- cur: 4 crf: 4 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 ug4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] cix: 1230 mrk:-- cur: 5 crf: 5 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 ug4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] cix: 1230 mrk:-- cur: 6 crf: 6 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 ug4/ok:( 0.438019574 0.438019574 ) 
    2017-12-17 14:16:19.432 INFO  [1197696] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 14:16:19.432 INFO  [1197696] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 14:16:19.433 ERROR [1197696] [CRandomEngine::postStep@323] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero YES
    2017-12-17 14:16:19.433 FATAL [1197696] [CRandomEngine::postStep@331] CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero 
    CRandomEngine_cc_postStep.[01] step_id:1 bst:   StepTooSmall pri:FresnelReflection okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    2017-12-17 14:16:19.436 INFO  [1197696] [CRandomEngine::flat@225]  --dbgkludgeflatzero   first flat call following boundary status StepTooSmall after FresnelReflection yields  _peek(-2) value  v 0.753801
    CRandomEngine_cc_flat.[07] cix: 1230 mrk:*# cur: 6 crf: 7 csf: 0 lg4/ok: (                     OpBoundary                   OpAbsorption ) df:       0.315781891 ug4/ok:( 0.753801465 0.438019574 ) 
    CRandomEngine_cc_flat.[08] cix: 1230 mrk:-- cur: 7 crf: 8 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.102905545e-10 ug4/ok:( 0.714031577 0.714031577 ) 
    CRandomEngine_cc_flat.[09] cix: 1230 mrk:-# cur: 8 crf: 9 csf: 2 lg4/ok: (                   OpAbsorption                     OpRayleigh ) df:   2.093353269e-10 ug4/ok:( 0.330403954 0.330403954 ) 
    CRandomEngine_cc_flat.[10] cix: 1230 mrk:-# cur: 9 crf:10 csf: 3 lg4/ok: ( OpBoundary_DiDiReflectOrTransmit                     OpRayleigh ) df:   4.423827971e-10 ug4/ok:( 0.570741653 0.570741653 ) 
    CRandomEngine_cc_flat.[11] cix: 1230 mrk:-# cur:10 crf:11 csf: 4 lg4/ok: (        OpBoundary_DoAbsorption                     OpRayleigh ) df:   1.903991964e-10 ug4/ok:( 0.375908673 0.375908673 ) 
    CRandomEngine_cc_postStep.[02] step_id:2 bst:     Absorption pri:   StepTooSmall okevt_pt:TO BR [SC] BT BR BT SA                             
    --


fixed  1230 : notice that this photon has two zeroSteps only the first gets dbgkludgeflatzero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero

    2017-12-17 14:09:42.271 INFO  [1195832] [CRandomEngine::preTrack@396] CRandomEngine::preTrack : DONE cmd "ucf.py 1230"
    CRandomEngine_cc_preTrack.[00] lucf:29 pindex:1230
    2017-12-17 14:09:42.285 ERROR [1195832] [CRandomEngine::preTrack@406] CRandomEngine::pretrack record_id:  ctx.record_id 0 use_index 1230 align_mask YES
    CRandomEngine_cc_flat.[00] cix: 1230 mrk:-- cur: 0 crf: 0 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   5.052794121e-14 ug4/ok:( 0.001117025 0.001117025 ) 
    CRandomEngine_cc_flat.[01] cix: 1230 mrk:-- cur: 1 crf: 1 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.976989766e-10 ug4/ok:( 0.502647340 0.502647340 ) 
    CRandomEngine_cc_flat.[02] cix: 1230 mrk:-- cur: 2 crf: 2 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   5.276490356e-11 ug4/ok:( 0.601504147 0.601504147 ) 
    CRandomEngine_cc_flat.[03] cix: 1230 mrk:-- cur: 3 crf: 3 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   3.701783324e-11 ug4/ok:( 0.938713491 0.938713491 ) 
    CRandomEngine_cc_postStep.[00] step_id:0 bst:FresnelReflection pri:      Undefined okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[04] cix: 1230 mrk:-- cur: 4 crf: 4 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 ug4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] cix: 1230 mrk:-- cur: 5 crf: 5 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 ug4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] cix: 1230 mrk:-- cur: 6 crf: 6 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 ug4/ok:( 0.438019574 0.438019574 ) 
    2017-12-17 14:09:42.325 INFO  [1195832] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 14:09:42.325 INFO  [1195832] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 14:09:42.326 ERROR [1195832] [CRandomEngine::postStep@323] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero YES
    2017-12-17 14:09:42.326 FATAL [1195832] [CRandomEngine::postStep@331] CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero 
    CRandomEngine_cc_postStep.[01] step_id:1 bst:   StepTooSmall pri:FresnelReflection okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    2017-12-17 14:09:42.329 ERROR [1195832] [CSteppingAction::UserSteppingAction@120]  --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft 
    2017-12-17 14:09:42.329 INFO  [1195832] [CRandomEngine::flat@225]  --dbgkludgeflatzero   first flat call following boundary status StepTooSmall after FresnelReflection yields  _peek(-2) value  v 0.753801
    CRandomEngine_cc_flat.[07] cix: 1230 mrk:*# cur: 6 crf: 7 csf: 0 lg4/ok: (                     OpBoundary                   OpAbsorption ) df:       0.315781891 ug4/ok:( 0.753801465 0.438019574 ) 
    CRandomEngine_cc_flat.[08] cix: 1230 mrk:-- cur: 7 crf: 8 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.102905545e-10 ug4/ok:( 0.714031577 0.714031577 ) 
    CRandomEngine_cc_flat.[09] cix: 1230 mrk:-- cur: 8 crf: 9 csf: 2 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.093353269e-10 ug4/ok:( 0.330403954 0.330403954 ) 
    CRandomEngine_cc_flat.[10] cix: 1230 mrk:-- cur: 9 crf:10 csf: 3 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.423827971e-10 ug4/ok:( 0.570741653 0.570741653 ) 
    CRandomEngine_cc_flat.[11] cix: 1230 mrk:-- cur:10 crf:11 csf: 4 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.903991964e-10 ug4/ok:( 0.375908673 0.375908673 ) 
    CRandomEngine_cc_flat.[12] cix: 1230 mrk:-- cur:11 crf:12 csf: 5 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.353455126e-10 ug4/ok:( 0.784978330 0.784978330 ) 
    CRandomEngine_cc_postStep.[02] step_id:2 bst:  NotAtBoundary pri:   StepTooSmall okevt_pt:TO BR [SC] BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[13] cix: 1230 mrk:-- cur:12 crf:13 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.406677163e-10 ug4/ok:( 0.892654359 0.892654359 ) 
    CRandomEngine_cc_flat.[14] cix: 1230 mrk:-- cur:13 crf:14 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.669952203e-10 ug4/ok:( 0.441063195 0.441063195 ) 
    CRandomEngine_cc_flat.[15] cix: 1230 mrk:-- cur:14 crf:15 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.626708933e-10 ug4/ok:( 0.773742437 0.773742437 ) 
    CRandomEngine_cc_flat.[16] cix: 1230 mrk:-- cur:15 crf:16 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   4.671020237e-10 ug4/ok:( 0.556839108 0.556839108 ) 
    CRandomEngine_cc_postStep.[03] step_id:3 bst:FresnelRefraction pri:  NotAtBoundary okevt_pt:TO BR SC [BT] BR BT SA                             
    --
    CRandomEngine_cc_flat.[17] cix: 1230 mrk:-- cur:16 crf:17 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:    1.88293825e-11 ug4/ok:( 0.775349319 0.775349319 ) 
    CRandomEngine_cc_flat.[18] cix: 1230 mrk:-- cur:17 crf:18 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.589111148e-10 ug4/ok:( 0.752141237 0.752141237 ) 
    CRandomEngine_cc_flat.[19] cix: 1230 mrk:-- cur:18 crf:19 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.373718527e-10 ug4/ok:( 0.412002385 0.412002385 ) 
    CRandomEngine_cc_postStep.[04] step_id:4 bst:TotalInternalReflection pri:FresnelRefraction okevt_pt:TO BR SC BT [BR] BT SA                             
    --
    CRandomEngine_cc_flat.[20] cix: 1230 mrk:-# cur:19 crf:20 csf: 0 lg4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:   4.672088827e-10 ug4/ok:( 0.282463104 0.282463104 ) 
    CRandomEngine_cc_flat.[21] cix: 1230 mrk:-# cur:20 crf:21 csf: 1 lg4/ok: (                     OpRayleigh                     OpBoundary ) df:   1.872253463e-10 ug4/ok:( 0.432497680 0.432497680 ) 
    CRandomEngine_cc_flat.[22] cix: 1230 mrk:-# cur:21 crf:22 csf: 2 lg4/ok: (                   OpAbsorption                     OpRayleigh ) df:   4.039001356e-10 ug4/ok:( 0.907848895 0.907848895 ) 
    2017-12-17 14:09:42.403 INFO  [1195832] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 14:09:42.403 INFO  [1195832] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 14:09:42.403 ERROR [1195832] [CRandomEngine::postStep@323] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero YES
    2017-12-17 14:09:42.403 FATAL [1195832] [CRandomEngine::postStep@331] CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero 
    CRandomEngine_cc_postStep.[05] step_id:5 bst:   StepTooSmall pri:TotalInternalReflection okevt_pt:TO BR SC BT [BR] BT SA                             
    --
    2017-12-17 14:09:42.407 ERROR [1195832] [CSteppingAction::UserSteppingAction@120]  --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft 
    CRandomEngine_cc_flat.[23] cix: 1230 mrk:-# cur:22 crf:23 csf: 0 lg4/ok: (                     OpBoundary                   OpAbsorption ) df:   7.296752091e-11 ug4/ok:( 0.912139237 0.912139237 ) 
    CRandomEngine_cc_flat.[24] cix: 1230 mrk:-- cur:23 crf:24 csf: 1 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   8.567047072e-11 ug4/ok:( 0.201808557 0.201808557 ) 
    CRandomEngine_cc_postStep.[06] step_id:6 bst:FresnelRefraction pri:   StepTooSmall okevt_pt:TO BR SC BT BR [BT] SA                             
    --
    CRandomEngine_cc_flat.[25] cix: 1230 mrk:-- cur:24 crf:25 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   4.876709037e-10 ug4/ok:( 0.795349360 0.795349360 ) 
    CRandomEngine_cc_flat.[26] cix: 1230 mrk:-- cur:25 crf:26 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.741393779e-10 ug4/ok:( 0.484203994 0.484203994 ) 
    CRandomEngine_cc_flat.[27] cix: 1230 mrk:-- cur:26 crf:27 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   4.411544741e-11 ug4/ok:( 0.093548603 0.093548603 ) 
    CRandomEngine_cc_flat.[28] cix: 1230 mrk:-- cur:27 crf:28 csf: 3 lg4/ok: ( OpBoundary_DiDiReflectOrTransmit OpBoundary_DiDiReflectOrTransmit ) df:    4.29260294e-10 ug4/ok:( 0.750533462 0.750533462 ) 
    CRandomEngine_cc_flat.[29] cix: 1230 mrk:-- cur:28 crf:29 csf: 4 lg4/ok: (        OpBoundary_DoAbsorption        OpBoundary_DoAbsorption ) df:   3.650513225e-10 ug4/ok:( 0.946246266 0.946246266 ) 
    CRandomEngine_cc_postStep.[07] step_id:7 bst:     Absorption pri:FresnelRefraction okevt_pt:TO BR SC BT BR BT [SA]                             
    --
    //                                                  CRandomEngine_cc_postTrack.[00] : postTrack label 
    CRandomEngine_cc_postTrack.[00] pindex:1230
    2017-12-17 14:09:42.444 INFO  [1195832] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1



broken correspondence
~~~~~~~~~~~~~~~~~~~~~~

::

    2017-12-17 13:48:32.453 INFO  [1189891] [CRandomEngine::preTrack@384] CRandomEngine::preTrack : DONE cmd "ucf.py 1230"
    CRandomEngine_cc_preTrack.[00] lucf:29 pindex:1230
    2017-12-17 13:48:32.466 ERROR [1189891] [CRandomEngine::preTrack@394] CRandomEngine::pretrack record_id:  ctx.record_id 0 use_index 1230 align_mask YES
    CRandomEngine_cc_flat.[00] cix: 1230 mrk:-- cur: 0 crf: 0 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   5.052794121e-14 ug4/ok:( 0.001117025 0.001117025 ) 
    CRandomEngine_cc_flat.[01] cix: 1230 mrk:-- cur: 1 crf: 1 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.976989766e-10 ug4/ok:( 0.502647340 0.502647340 ) 
    CRandomEngine_cc_flat.[02] cix: 1230 mrk:-- cur: 2 crf: 2 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   5.276490356e-11 ug4/ok:( 0.601504147 0.601504147 ) 
    CRandomEngine_cc_flat.[03] cix: 1230 mrk:-- cur: 3 crf: 3 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   3.701783324e-11 ug4/ok:( 0.938713491 0.938713491 ) 
    CRandomEngine_cc_postStep.[00] step_id:0 bst:FresnelReflection pri:      Undefined okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[04] cix: 1230 mrk:-- cur: 4 crf: 4 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 ug4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] cix: 1230 mrk:-- cur: 5 crf: 5 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 ug4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] cix: 1230 mrk:-- cur: 6 crf: 6 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 ug4/ok:( 0.438019574 0.438019574 ) 
    2017-12-17 13:48:32.506 INFO  [1189891] [*DsG4OpBoundaryProcess::PostStepDoIt@247]  StepTooSmall
    2017-12-17 13:48:32.506 INFO  [1189891] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-17 13:48:32.506 ERROR [1189891] [CRandomEngine::postStep@311] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojumpzero YES
    2017-12-17 13:48:32.506 FATAL [1189891] [CRandomEngine::postStep@319] CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero 
    CRandomEngine_cc_postStep.[01] step_id:1 bst:   StepTooSmall pri:FresnelReflection okevt_pt:TO [BR] SC BT BR BT SA                             
    --
    2017-12-17 13:48:32.510 ERROR [1189891] [CSteppingAction::UserSteppingAction@120]  --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft 
    2017-12-17 13:48:32.510 INFO  [1189891] [CRandomEngine::flat@225]  --dbgkludgeflatzero   first flat call following boundary status StepTooSmall after FresnelReflection yields  _peek(-3) value  v 0.938713
    CRandomEngine_cc_flat.[07] cix: 1230 mrk:*# cur: 6 crf: 7 csf: 0 lg4/ok: (                     OpBoundary                   OpAbsorption ) df:       0.500693917 ug4/ok:( 0.938713491 0.438019574 ) 
    CRandomEngine_cc_flat.[08] cix: 1230 mrk:-- cur: 7 crf: 8 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.102905545e-10 ug4/ok:( 0.714031577 0.714031577 ) 
    CRandomEngine_cc_flat.[09] cix: 1230 mrk:-- cur: 8 crf: 9 csf: 2 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.093353269e-10 ug4/ok:( 0.330403954 0.330403954 ) 
    CRandomEngine_cc_flat.[10] cix: 1230 mrk:-- cur: 9 crf:10 csf: 3 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.423827971e-10 ug4/ok:( 0.570741653 0.570741653 ) 
    CRandomEngine_cc_flat.[11] cix: 1230 mrk:-- cur:10 crf:11 csf: 4 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.903991964e-10 ug4/ok:( 0.375908673 0.375908673 ) 
    CRandomEngine_cc_flat.[12] cix: 1230 mrk:-- cur:11 crf:12 csf: 5 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.353455126e-10 ug4/ok:( 0.784978330 0.784978330 ) 
    CRandomEngine_cc_postStep.[02] step_id:2 bst:  NotAtBoundary pri:   StepTooSmall okevt_pt:TO BR [SC] BT BR BT SA                             
    --
    CRandomEngine_cc_flat.[13] cix: 1230 mrk:-- cur:12 crf:13 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:   3.406677163e-10 ug4/ok:( 0.892654359 0.892654359 ) 
    CRandomEngine_cc_flat.[14] cix: 1230 mrk:-- cur:13 crf:14 csf: 1 lg4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.669952203e-10 ug4/ok:( 0.441063195 0.441063195 ) 
    CRandomEngine_cc_flat.[15] cix: 1230 mrk:-- cur:14 crf:15 csf: 2 lg4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.626708933e-10 ug4/ok:( 0.773742437 0.773742437 ) 
    CRandomEngine_cc_flat.[16] cix: 1230 mrk:-- cur:15 crf:16 csf: 3 lg4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   4.671020237e-10 ug4/ok:( 0.556839108 0.556839108 ) 
    CRandomEngine_cc_postStep.[03] step_id:3 bst:FresnelRefraction pri:  NotAtBoundary okevt_pt:TO BR SC [BT] BR BT SA                             
    --
    CRandomEngine_cc_flat.[17] cix: 1230 mrk:-- cur:16 crf:17 csf: 0 lg4/ok: (                     OpBoundary                     OpBoundary ) df:    1.88293825e-11 ug4/ok:( 0.775349319 0.77534931




kludge breaks the python debug comparison : seqs appear offset by 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    CRandomEngine_cc_flat.[26] mrk:*# crf:26 csf: 1 loc_g4/ok: (                     OpRayleigh                   OpAbsorption ) df:      0.1823485936 u_g4/ok:( 0.237027600 0.419376194 ) 
    CRandomEngine_cc_flat.[27] mrk:*# crf:27 csf: 2 loc_g4/ok: (                   OpAbsorption OpBoundary_DiDiReflectOrTransmit ) df:    0.005040466477 u_g4/ok:( 0.419376194 0.414335728 ) 
    G4SteppingManager_cc_191.[07] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    CRandomEngine_cc_flat.[28] mrk:*# crf:28 csf: 3 loc_g4/ok: ( OpBoundary_DiDiReflectOrTransmit        OpBoundary_DoAbsorption ) df:     0.09838336669 u_g4/ok:( 0.414335728 0.315952361 ) 
    CRandomEngine_cc_flat.[29] mrk:*# crf:29 csf: 4 loc_g4/ok: (        OpBoundary_DoAbsorption                   ucf-overflow ) df:       1.315952361 u_g4/ok:( 0.315952361 -1.000000000 ) 
    CRec_cc_add.[07] : bst:          Absorption pri:   FresnelRefraction :  
    CRandomEngine_cc_postStep.[07] step_id:7 okevt_pt:   



what happens with full run with triple kludge
--------------------------------------------------

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero



devious kludge working to some extent  --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero
------------------------------------------------------------------------------------------------


running the kludge
~~~~~~~~~~~~~~~~~~~~

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero
          YEP
    tboolean-;tboolean-box --okg4 --align --mask 9041 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero 
          YEP
    tboolean-;tboolean-box --okg4 --align --mask 14510 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero 
          YEP
    tboolean-;tboolean-box --okg4 --align --mask 49786 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero 
          YEP
    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero 
          YEP
    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero 
          YEP

    tboolean-;tboolean-box --okg4 --align --mask 1230,9041,14510,49786,69653,77962 --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero 



dbgskipclearzero 
~~~~~~~~~~~~~~~~~~~~~

Prevents the end of step OpRayleigh + OpAbsorption interaction length clear, so the 
next step RNG consumption for those processes is not done, leaving just OpBoundary consumption.


::

    115         bool zeroStep = m_ctx._noZeroSteps > 0 ;   // usually means there was a jump back 
    116         bool skipClear = zeroStep && m_ok->isDbgSkipClearZero()  ;
    117 
    118         if(skipClear)
    119         {
    120             LOG(error) << " --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft " ;
    121         }
    122         else
    123         {
    124             CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx._process_manager, *m_ctx._track, *m_ctx._step );
    125         }
    126 

    delta:cfg4 blyth$ grep dbgskipclearzero *.*
    CSteppingAction.cc:            LOG(error) << " --dbgskipclearzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft " ; 


dbgnojumpzero
~~~~~~~~~~~~~~~~
    
Zero steps burn 3 RNG in the decision making, normally alignment is retained by 
rewinding the sequence. Which means that when G4 gets over the zero step it
will come up with the same decision again, as Opticks did already. 

Inhibiting this is probably something that only works for the 
6 maligned ? 

::

    296 // invoked by CG4::postStep
    297 void CRandomEngine::postStep()
    298 {
    299     if(m_ctx._noZeroSteps > 0)
    300     {
    302         int backseq = -m_current_step_flat_count ;
    303         bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ;
    304 
    305         LOG(error) << "CRandomEngine::postStep"
    306                    << " _noZeroSteps " << m_ctx._noZeroSteps
    307                    << " backseq " << backseq
    308                    << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
    309                    ;
    310 
    311         if( dbgnojumpzero )
    312         {
    313             LOG(fatal) << "CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero " ;
    314         }
    315         else
    316         {
    317             jump(backseq);
    318         }
    319     }


    delta:cfg4 blyth$ grep dbgnojumpzero *.*
    CRandomEngine.cc:        bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ; 
    CRandomEngine.cc:                   << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
    CRandomEngine.cc:        if( dbgnojumpzero )
    CRandomEngine.cc:            LOG(fatal) << "CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero " ;   



dbgkludgeflatzero
~~~~~~~~~~~~~~~~~~~

::

    209 double CRandomEngine::flat()
    210 {       
    211     if(!m_internal) m_location = CurrentProcessName();
    212     assert( m_current_record_flat_count < m_curand_nv ); 
    213     
    214     bool kludge = m_dbgkludgeflatzero 
    215                && m_current_step_flat_count == 0
    216                && m_ctx._boundary_status == StepTooSmall
    217                && m_ctx._prior_boundary_status == FresnelReflection   
    218                ;
    219                 
    220     double v = kludge ? _peek(-3) : _flat() ; 
    221     
    222     if( kludge )
    223     {
    224         LOG(info) << " --dbgkludgeflatzero  "
    225                   << " first flat call following FresnelReflection then StepTooSmall yields  _peek(-3) value "
    226                   << " v " << v 
    227                  ;
    228     }            
    229     
    230     m_flat = v ; 
    231     
    232     m_current_record_flat_count++ ;  // (*lldb*) flat 
    233     m_current_step_flat_count++ ;
    234     
    235     return m_flat ;
    236 }   


    delta:cfg4 blyth$ grep dbgkludgeflatzero *.*
    CRandomEngine.cc:    m_dbgkludgeflatzero(m_ok->isDbgKludgeFlatZero()), 
    CRandomEngine.cc:    bool kludge = m_dbgkludgeflatzero 
    CRandomEngine.cc:        LOG(info) << " --dbgkludgeflatzero  "
    CRandomEngine.hh:        bool                          m_dbgkludgeflatzero ; 


With the triple whammy kludge the six get perfectly aligned
-------------------------------------------------------------

::

    In [3]: ab.a.rpost_(slice(0,10))
    Out[3]: 
    A()sliced
    A([[[ -37.8781,   11.8231, -449.8989,    0.2002],
        [ -37.8781,   11.8231,  -99.9944,    1.3672],
        [ -37.8781,   11.8231, -253.2548,    1.8781],
        [  97.7921,  -52.7844,  -99.9944,    2.5941],
        [ 149.9984,  -77.6556,   24.307 ,    3.4248],
        [ 118.2039,  -92.7959,   99.9944,    3.9308],
        [-191.6203, -240.3581,  449.9952,    5.566 ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]],

       [[  34.0518,  -32.3038, -449.8989,    0.2002],
        [  34.0518,  -32.3038,  -99.9944,    1.3672],
        [  34.0518,  -32.3038,   51.3529,    2.284 ],
        [-149.9984,   23.4261,  -20.4256,    3.5279],


    In [4]: ab.a.rpost_(slice(0,10)).shape
    Out[4]: (6, 10, 4)

    In [5]: ab.b.rpost_(slice(0,10)).shape
    Out[5]: (6, 10, 4)

    In [6]: dv = ab.a.rpost_(slice(0,10)) - ab.b.rpost_(slice(0,10))
    Out[6]: 
    A()sliced
    A([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],

    In [7]: dv = ab.a.rpost_(slice(0,10)) - ab.b.rpost_(slice(0,10))

    In [8]: dv.max()
    Out[8]: 
    A()sliced
    A(0.0)

    In [9]: dv = ab.a.rpolw_(slice(0,10)) - ab.b.rpolw_(slice(0,10))

    In [10]: dv.max()
    Out[10]: 
    A()sliced
    A(0.0, dtype=float32)






Review Rewinding
------------------

Rewinding noted in :doc:`BR_PhysicalStep_zero_misalignment`

::

    Smouldering evidence : PhysicalStep-zero/StepTooSmall results in RNG mis-alignment 
    ------------------------------------------------------------------------------------

    Some G4 technicality yields zero step at BR, that means the lucky scatter 
    throw that Opticks saw was not seen by G4 : as the sequence gets out of alignment.


Zero steps result in G4 burning an entire steps RNGs compared to Opticks.  
The solution was to jump back in the sequence on the G4 side.
However for the misaligned six (the 3~4 studied) all appear to have an improper
jump back.


::

    231 void CRandomEngine::poststep()
    232 {
    233     if(m_ctx._noZeroSteps > 0)
    234     {
    235         int backseq = -m_current_step_flat_count ;
    236         LOG(error) << "CRandomEngine::poststep"
    237                    << " _noZeroSteps " << m_ctx._noZeroSteps
    238                    << " backseq " << backseq
    239                    ;
    240         jump(backseq);
    241     }
    242 
    243     m_current_step_flat_count = 0 ;
    244 
    245     if( m_locseq )
    246     {
    247         m_locseq->poststep();
    248         LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ;
    249     }
    250 }


Review POstStep ClearNumberOfInteractionLengthLeft
------------------------------------------------------

At the end of everystep the RNG for AB and SC are cleared, in order to 
force G4VProcess::ResetNumberOfInteractionLengthLeft for every step, as
that is how Opticks works with AB and SC RNG consumption at every "propagate_to_boundary".

* hmm is OpBoundary skipped because its the winner process ? 
  so the standard G4VDiscreteProcess::PostStepDoIt will do the RNG consumption without assistance ?

See :doc:`stepping_process_review`

::

     59 /*
     60 
     61      95 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     62      96 {
     63      97   theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
     64      98   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
     65      99 }
     66 
     67 */
     68 
     69 
     70 void CProcessManager::ClearNumberOfInteractionLengthLeft(G4ProcessManager* proMgr, const G4Track& aTrack, const G4Step& aStep)
     71 {
     72     G4ProcessVector* pl = proMgr->GetProcessList() ;
     73     G4int n = pl->entries() ;
     74 
     75     for(int i=0 ; i < n ; i++)
     76     {
     77         G4VProcess* p = (*pl)[i] ;
     78         const G4String& name = p->GetProcessName() ;
     79         bool is_ab = name.compare("OpAbsorption") == 0 ;
     80         bool is_sc = name.compare("OpRayleigh") == 0 ;
     81         //bool is_bd = name.compare("OpBoundary") == 0 ;
     82         if( is_ab || is_sc )
     83         {
     84             G4VDiscreteProcess* dp = dynamic_cast<G4VDiscreteProcess*>(p) ;
     85             assert(dp);   // Transportation not discrete
     86             dp->G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );
     87             // devious way to invoke the protected ClearNumberOfInteractionLengthLeft via G4VDiscreteProcess::PostStepDoIt
     88         }
     89     }
     90 }







Arriving at the kludge
--------------------------


1230 : g4 wants to start again, but opticks was to scatter (bst:        StepTooSmall pri:   FresnelReflection :)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* interaction length decision consumes 3 (OpBoundary, OpRayleigh, OpAbsorption)
* one turn of scatter do loop consumes 5 (OpRayleigh)

* the post "StepToSmall" aka zero-step trick of G4 rewind -3, looks like it 
  does not work when StepTooSmall follows on from FresnelReflection

  * the -3 rewind feeds G4 the same RNG next, so it can makes the same decision   

  * actually it looks like rewinding -6 might work  : it didnt 



::

    CRandomEngine_cc_postStep.[00] step_id:0 okevt_pt:BR 
    CRandomEngine_cc_flat.[04] mrk:-- crf: 4 csf: 0 loc_g4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 u_g4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] mrk:-- crf: 5 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 u_g4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] mrk:-- crf: 6 csf: 2 loc_g4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 u_g4/ok:( 0.438019574 0.438019574 ) 
    G4SteppingManager_cc_191.[01] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    2017-12-16 14:42:20.051 INFO  [1012816] [CSteppingAction::setStep@148]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    CRec_cc_add.[01] : bst:        StepTooSmall pri:   FresnelReflection :  
    2017-12-16 14:42:20.054 ERROR [1012816] [CRandomEngine::postStep@279] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojump NO
    CRandomEngine_cc_jump.[00] cursor_old:7 jump_:-3 jump_count:1 cursor:4 
    CRandomEngine_cc_postStep.[01] step_id:1 okevt_pt:SC 

    CRandomEngine_cc_flat.[07] mrk:*# crf: 7 csf: 0 loc_g4/ok: (                     OpBoundary                     OpRayleigh ) df:     0.03976988803 u_g4/ok:( 0.753801465 0.714031577 ) 
    CRandomEngine_cc_flat.[08] mrk:*- crf: 8 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:      0.6694428025 u_g4/ok:( 0.999846756 0.330403954 ) 
    CRandomEngine_cc_flat.[09] mrk:*# crf: 9 csf: 2 loc_g4/ok: (                   OpAbsorption                     OpRayleigh ) df:      0.1327220793 u_g4/ok:( 0.438019574 0.570741653 ) 
    G4SteppingManager_cc_191.[02] :    fPostStepDoItProc : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 

    CRandomEngine_cc_flat.[10] mrk:*- crf:10 csf: 3 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:      0.3381229041 u_g4/ok:( 0.714031577 0.375908673 ) 
    CRandomEngine_cc_flat.[11] mrk:*- crf:11 csf: 4 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:      0.4545743762 u_g4/ok:( 0.330403954 0.784978330 ) 
    CRandomEngine_cc_flat.[12] mrk:*# crf:12 csf: 5 loc_g4/ok: (                     OpRayleigh                     OpBoundary ) df:      0.3219127056 u_g4/ok:( 0.570741653 0.892654359 ) 
    CRandomEngine_cc_flat.[13] mrk:*- crf:13 csf: 6 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:     0.06515452219 u_g4/ok:( 0.375908673 0.441063195 ) 
    CRandomEngine_cc_flat.[14] mrk:*# crf:14 csf: 7 loc_g4/ok: (                     OpRayleigh                   OpAbsorption ) df:     0.01123589314 u_g4/ok:( 0.784978330 0.773742437 ) 

    CRec_cc_add.[02] : bst:       NotAtBoundary pri:        StepTooSmall :  
    CRandomEngine_cc_postStep.[02] step_id:2 okevt_pt:BT 
    CRandomEngine_cc_flat.[15] mrk:*# crf:15 csf: 0 loc_g4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:      0.3358152513 u_g4/ok:( 0.892654359 0.556839108 ) 
    CRandomEngine_cc_flat.[16] mrk:*# crf:16 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpBoundary ) df:      0.3342861235 u_g4/ok:( 0.441063195 0.775349319 ) 
    CRandomEngine_cc_flat.[17] mrk:*# crf:17 csf: 2 loc_g4/ok: (                   OpAbsorption                     OpRayleigh ) df:     0.02160120036 u_g4/ok:( 0.773742437 0.752141237 ) 






1230 : --dbgnojumpzero --dbgskipclearzero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* not rewinding and clearing after zero-step gets close, just have to pursuade OpBoundary not to throw again
  despite it being the process


::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog  -DD --dbgskipclearzero --dbgnojumpzero
 

    RandomEngine_cc_flat.[06] mrk:-- crf: 6 csf: 2 loc_g4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 u_g4/ok:( 0.438019574 0.438019574 ) 
    G4SteppingManager_cc_191.[01] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    2017-12-16 16:11:03.804 INFO  [1038396] [CSteppingAction::setStep@159]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    CRec_cc_add.[01] : bst:        StepTooSmall pri:   FresnelReflection :  
    2017-12-16 16:11:03.807 ERROR [1038396] [CRandomEngine::postStep@280] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojump YES
    2017-12-16 16:11:03.807 FATAL [1038396] [CRandomEngine::postStep@288] CRandomEngine::postStep rewind inhibited by option: --dbgnojump 
    CRandomEngine_cc_postStep.[01] step_id:1 okevt_pt:SC 
    2017-12-16 16:11:03.810 ERROR [1038396] [CSteppingAction::UserSteppingAction@120]  --dbgskipclearafterzero  skipping CProcessManager::ClearNumberOfInteractionLengthLeft 
    CRandomEngine_cc_flat.[07] mrk:-# crf: 7 csf: 0 loc_g4/ok: (                     OpBoundary                     OpRayleigh ) df:   1.102905545e-10 u_g4/ok:( 0.714031577 0.714031577 ) 
    G4SteppingManager_cc_191.[02] :    fPostStepDoItProc : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    CRandomEngine_cc_flat.[08] mrk:-- crf: 8 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.093353269e-10 u_g4/ok:( 0.330403954 0.330403954 ) 
    CRandomEngine_cc_flat.[09] mrk:-- crf: 9 csf: 2 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   4.423827971e-10 u_g4/ok:( 0.570741653 0.570741653 ) 
    CRandomEngine_cc_flat.[10] mrk:-- crf:10 csf: 3 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.903991964e-10 u_g4/ok:( 0.375908673 0.375908673 ) 
    CRandomEngine_cc_flat.[11] mrk:-- crf:11 csf: 4 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   1.353455126e-10 u_g4/ok:( 0.784978330 0.784978330 ) 
    CRandomEngine_cc_flat.[12] mrk:-# crf:12 csf: 5 loc_g4/ok: (                     OpRayleigh                     OpBoundary ) df:   3.406677163e-10 u_g4/ok:( 0.892654359 0.892654359 ) 




1230 : trying a jump back of -6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    2017-12-16 15:20:47.342 INFO  [1023297] [SSys::run@50] ucf.py 1230 rc_raw : 0 rc : 0
    2017-12-16 15:20:47.343 INFO  [1023297] [CRandomEngine::preTrack@345] CRandomEngine::preTrack : DONE cmd "ucf.py 1230"
    CRandomEngine_cc_preTrack.[00] lucf:29 pindex:1230
    2017-12-16 15:20:47.356 ERROR [1023297] [CRandomEngine::preTrack@354] CRandomEngine::pretrack record_id:  ctx.record_id 0 use_index 1230 with_mask YES
    CRandomEngine_cc_flat.[00] mrk:-- crf: 0 csf: 0 loc_g4/ok: (                     OpBoundary                     OpBoundary ) df:   5.052794121e-14 u_g4/ok:( 0.001117025 0.001117025 ) 
    CRandomEngine_cc_flat.[01] mrk:-- crf: 1 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.976989766e-10 u_g4/ok:( 0.502647340 0.502647340 ) 
    CRandomEngine_cc_flat.[02] mrk:-- crf: 2 csf: 2 loc_g4/ok: (                   OpAbsorption                   OpAbsorption ) df:   5.276490356e-11 u_g4/ok:( 0.601504147 0.601504147 ) 
    G4SteppingManager_cc_191.[00] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    CRandomEngine_cc_flat.[03] mrk:-- crf: 3 csf: 3 loc_g4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   3.701783324e-11 u_g4/ok:( 0.938713491 0.938713491 ) 
    CRec_cc_add.[00] : bst:   FresnelReflection pri:           Undefined :  
    CRandomEngine_cc_postStep.[00] step_id:0 okevt_pt:BR 
    CRandomEngine_cc_flat.[04] mrk:-- crf: 4 csf: 0 loc_g4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 u_g4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] mrk:-- crf: 5 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 u_g4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] mrk:-- crf: 6 csf: 2 loc_g4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 u_g4/ok:( 0.438019574 0.438019574 ) 
    G4SteppingManager_cc_191.[01] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    2017-12-16 15:20:47.825 INFO  [1023297] [CSteppingAction::setStep@155]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    CRec_cc_add.[01] : bst:        StepTooSmall pri:   FresnelReflection :  
    2017-12-16 15:20:47.828 ERROR [1023297] [CRandomEngine::postStep@279] CRandomEngine::postStep _noZeroSteps 1 backseq -6 --dbgnojump NO
    CRandomEngine_cc_jump.[00] cursor_old:7 jump_:-6 jump_count:1 cursor:1 
    CRandomEngine_cc_postStep.[01] step_id:1 okevt_pt:SC 
    CRandomEngine_cc_flat.[07] mrk:*# crf: 7 csf: 0 loc_g4/ok: (                     OpBoundary                     OpRayleigh ) df:      0.2113842367 u_g4/ok:( 0.502647340 0.714031577 ) 
    CRandomEngine_cc_flat.[08] mrk:*- crf: 8 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:      0.2711001931 u_g4/ok:( 0.601504147 0.330403954 ) 
    CRandomEngine_cc_flat.[09] mrk:*# crf: 9 csf: 2 loc_g4/ok: (                   OpAbsorption                     OpRayleigh ) df:       0.367971838 u_g4/ok:( 0.938713491 0.570741653 ) 
    G4SteppingManager_cc_191.[02] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    CRandomEngine_cc_flat.[10] mrk:*# crf:10 csf: 3 loc_g4/ok: ( OpBoundary_DiDiReflectOrTransmit                     OpRayleigh ) df:       0.377892792 u_g4/ok:( 0.753801465 0.375908673 ) 
    CRandomEngine_cc_flat.[11] mrk:*# crf:11 csf: 4 loc_g4/ok: (        OpBoundary_DoAbsorption                     OpRayleigh ) df:      0.2148684265 u_g4/ok:( 0.999846756 0.784978330 ) 
    CRec_cc_add.[02] : bst:          Absorption pri:        StepTooSmall :  
    CRandomEngine_cc_postStep.[02] step_id:2 okevt_pt:BT 
    //                                                  CRandomEngine_cc_postTrack.[00] : postTrack label 
    CRandomEngine_cc_postTrack.[00] pindex:1230




1230 with rewind inhibited gets G4 to make different decision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    CRandomEngine_cc_flat.[03] mrk:-- crf: 3 csf: 3 loc_g4/ok: (      OpBoundary_DiDiTransCoeff      OpBoundary_DiDiTransCoeff ) df:   3.701783324e-11 u_g4/ok:( 0.938713491 0.938713491 ) 
    CRec_cc_add.[00] : bst:   FresnelReflection pri:           Undefined :  
    CRandomEngine_cc_postStep.[00] step_id:0 okevt_pt:BR 
    CRandomEngine_cc_flat.[04] mrk:-- crf: 4 csf: 0 loc_g4/ok: (                     OpBoundary                     OpBoundary ) df:   3.448485941e-11 u_g4/ok:( 0.753801465 0.753801465 ) 
    CRandomEngine_cc_flat.[05] mrk:-- crf: 5 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:    4.58282523e-10 u_g4/ok:( 0.999846756 0.999846756 ) 
    CRandomEngine_cc_flat.[06] mrk:-- crf: 6 csf: 2 loc_g4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.114929426e-10 u_g4/ok:( 0.438019574 0.438019574 ) 
    G4SteppingManager_cc_191.[01] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    2017-12-16 14:38:31.473 INFO  [1011620] [CSteppingAction::setStep@148]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    CRec_cc_add.[01] : bst:        StepTooSmall pri:   FresnelReflection :  
    2017-12-16 14:38:31.477 ERROR [1011620] [CRandomEngine::postStep@279] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojump YES
    2017-12-16 14:38:31.477 FATAL [1011620] [CRandomEngine::postStep@287] CRandomEngine::postStep rewind inhibited by option: --dbgnojump 
    CRandomEngine_cc_postStep.[01] step_id:1 okevt_pt:SC 
    CRandomEngine_cc_flat.[07] mrk:-# crf: 7 csf: 0 loc_g4/ok: (                     OpBoundary                     OpRayleigh ) df:   1.102905545e-10 u_g4/ok:( 0.714031577 0.714031577 ) 
    CRandomEngine_cc_flat.[08] mrk:-- crf: 8 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   2.093353269e-10 u_g4/ok:( 0.330403954 0.330403954 ) 
    CRandomEngine_cc_flat.[09] mrk:-# crf: 9 csf: 2 loc_g4/ok: (                   OpAbsorption                     OpRayleigh ) df:   4.423827971e-10 u_g4/ok:( 0.570741653 0.570741653 ) 
    G4SteppingManager_cc_191.[02] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    CRandomEngine_cc_flat.[10] mrk:-# crf:10 csf: 3 loc_g4/ok: ( OpBoundary_DiDiReflectOrTransmit                     OpRayleigh ) df:   1.903991964e-10 u_g4/ok:( 0.375908673 0.375908673 ) 
    CRandomEngine_cc_flat.[11] mrk:-# crf:11 csf: 4 loc_g4/ok: (        OpBoundary_DoAbsorption                     OpRayleigh ) df:   1.353455126e-10 u_g4/ok:( 0.784978330 0.784978330 ) 
    CRec_cc_add.[02] : bst:          Absorption pri:        StepTooSmall :  
    CRandomEngine_cc_postStep.[02] step_id:2 okevt_pt:BT 
    //                                                  CRandomEngine_cc_postTrack.[00] : postTrack label 
    CRandomEngine_cc_postTrack.[00] pindex:1230
    2017-12-16 14:38:31.510 INFO  [1011620] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2017-12-16 14:38:31.510 INFO  [1011620] [CG4::postpropagate@434] CG4::postpropagate(0) ctx CG4Ctx::desc_stats dump_count 0 event_total 1 event_track_count 1
    2017-12-16 14:38:31.510 INFO  [1011620] [OpticksEvent::postPropagateGeant4@2039] OpticksEvent::postPropagateGeant4 


9041
~~~~~~

::

    CRandomEngine_cc_postStep.[02] step_id:2 okevt_pt:BR 
    CRandomEngine_cc_flat.[21] mrk:-- crf:21 csf: 0 loc_g4/ok: (                     OpBoundary                     OpBoundary ) df:   9.005740598e-11 u_g4/ok:( 0.885444343 0.885444343 ) 
    CRandomEngine_cc_flat.[22] mrk:-- crf:22 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpRayleigh ) df:   3.500061352e-10 u_g4/ok:( 0.554676592 0.554676592 ) 
    CRandomEngine_cc_flat.[23] mrk:-- crf:23 csf: 2 loc_g4/ok: (                   OpAbsorption                   OpAbsorption ) df:   3.905334389e-10 u_g4/ok:( 0.302562296 0.302562296 ) 
    G4SteppingManager_cc_191.[03] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    2017-12-16 14:34:19.324 INFO  [1009771] [CSteppingAction::setStep@148]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    CRec_cc_add.[03] : bst:        StepTooSmall pri:   FresnelReflection :  
    2017-12-16 14:34:19.326 ERROR [1009771] [CRandomEngine::postStep@279] CRandomEngine::postStep _noZeroSteps 1 backseq -3 --dbgnojump NO
    CRandomEngine_cc_jump.[00] cursor_old:24 jump_:-3 jump_count:1 cursor:21 
    CRandomEngine_cc_postStep.[03] step_id:3 okevt_pt:BR 
    CRandomEngine_cc_flat.[24] mrk:*# crf:24 csf: 0 loc_g4/ok: (                     OpBoundary      OpBoundary_DiDiTransCoeff ) df:      0.3547135591 u_g4/ok:( 0.885444343 0.530730784 ) 
    CRandomEngine_cc_flat.[25] mrk:*# crf:25 csf: 1 loc_g4/ok: (                     OpRayleigh                     OpBoundary ) df:      0.1313142176 u_g4/ok:( 0.554676592 0.685990810 ) 
    CRandomEngine_cc_flat.[26] mrk:*# crf:26 csf: 2 loc_g4/ok: (                   OpAbsorption                     OpRayleigh ) df:      0.2992141846 u_g4/ok:( 0.302562296 0.601776481 ) 
    G4SteppingManager_cc_191.[04] :        fGeomBoundary : After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs() 
    CRec_cc_add.[04] : bst:TotalInternalReflection pri:        StepTooSmall :  
    CRandomEngine_cc_postStep.[04] step_id:4 okevt_pt:BR 



Are all photons with scatter SC and a jump maligned ? NO : 6/28 photons with jumps and scatters are misaligned 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [1]: from opticks.ana.nload import np_load

    In [2]: jp = np_load("$TMP/CRandomEngine_jump_photons.npy")

    In [4]: jp.shape
    Out[4]: (12137,)

    In [3]: ab.dumpline(jp[:100])   ## all have a BR
          0   9979 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          1   9978 :                                           TO BR SA                                           TO BR SA 
          2   9968 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          3   9963 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          4   9961 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          5   9939 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          6   9932 :                                           TO BR SA                                           TO BR SA 
          7   9927 :                                           TO BR SA                                           TO BR SA 
          8   9923 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
          9   9915 :                                           TO BR SA                                           TO BR SA 
         10   9914 :                                     TO BT BR BT SA                                     TO BT BR BT SA 
         11   9911 :                                           TO BR SA                                           TO BR SA 

    In [11]: ab.maligned
    Out[11]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [12]: map(lambda _:_ in jp, ab.maligned)
    Out[12]: [True, True, True, True, True, True]

    In [6]: ab.a.pflags_where("SC").shape
    Out[6]: (92,)

    In [4]: ab.dumpline(ab.a.pflags_where("SC"))
          0    420 :                                           TO SC SA                                           TO SC SA 
          1    595 :                                  TO SC BT BR BT SA                                  TO SC BT BR BT SA 
          2   1198 :                                           TO SC SA                                           TO SC SA 
          3   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          4   2413 :                         TO BT BT SC BT BR BR BT SA                         TO BT BT SC BT BR BR BT SA 
          5   2658 :                                           TO SC SA                                           TO SC SA 
          6   4608 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          7   4777 :                                     TO BT BT SC SA                                     TO BT BT SC SA 
          8   5113 :                                           TO SC SA                                           TO SC SA 
          9   5729 :                                     TO BT BT SC SA                                     TO BT BT SC SA 
         10   6058 :                                           TO SC SA                                           TO SC SA 
         11   7258 :                               TO BT BT SC BT BT SA                               TO BT BT SC BT BT SA 
         12   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 

    In [11]: from opticks.ana.seq import seq2msk

    In [16]: ab.a.hismask.code("SC")
    Out[16]: 32

    In [15]: jp[np.where( seq2msk(ab.a.seqhis[jp]) & 32 )]   ## finding jumps with a scatter 
    Out[15]: 
    array([ 9041,  2413,  1230,   595, 19361, 18921, 14747, 14510, 26635, 36621, 33262, 30272, 49786, 58609, 58189, 53964, 69653, 65850, 60803, 77962, 76467, 73241, 87674, 97887, 95722, 94891, 92353,
           90322], dtype=uint32)

    In [17]: jpsc = jp[np.where( seq2msk(ab.a.seqhis[jp]) & 32 )]

    In [1]: a_jpsc
    Out[1]: 
    array([ 9041,  2413,  1230,   595, 19361, 18921, 14747, 14510, 26635, 36621, 33262, 30272, 49786, 58609, 58189, 53964, 69653, 65850, 60803, 77962, 76467, 73241, 87674, 97887, 95722, 94891, 92353,
           90322], dtype=uint32)

    In [3]: np.all( a_jpsc == b_jpsc )
    Out[3]: True

    In [4]: ab.dumpline(a_jpsc)   ## 6 out of 28 photons with jumps and scatters are misaligned 

          0   9041 : * :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          1   2413 :   :                         TO BT BT SC BT BR BR BT SA                         TO BT BT SC BT BR BR BT SA 
          2   1230 : * :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          3    595 :   :                                  TO SC BT BR BT SA                                  TO SC BT BR BT SA 
          4  19361 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
          5  18921 :   :                                  TO BR SC BT BT SA                                  TO BR SC BT BT SA 
          6  14747 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
          7  14510 : * :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          8  26635 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
          9  36621 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         10  33262 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         11  30272 :   :                      TO BT BR SC BR BR BR BR BR BR                      TO BT BR SC BR BR BR BR BR BR 
         12  49786 : * :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
         13  58609 :   :                            TO BT BT SC BT BR BT SA                            TO BT BT SC BT BR BT SA 
         14  58189 :   :                                  TO SC BT BR BT SA                                  TO SC BT BR BT SA 
         15  53964 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         16  69653 : * :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
         17  65850 :   :                            TO BT BT SC BT BR BT SA                            TO BT BT SC BT BR BT SA 
         18  60803 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         19  77962 : * :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 
         20  76467 :   :                                  TO BT BR SC BT SA                                  TO BT BR SC BT SA 
         21  73241 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         22  87674 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA 
         23  97887 :   :                                  TO SC BT BR BT SA                                  TO SC BT BR BT SA 
         24  95722 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA 
         25  94891 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         26  92353 :   :                            TO BT BT SC BT BR BT SA                            TO BT BT SC BT BR BT SA 
         27  90322 :   :                                  TO BT BT SC BR SA                                  TO BT BT SC BR SA 

    In [5]: 





Location misaligns
-------------------


::

    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD  --dbgnojump


    [  6]                                       OpAbsorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

    2017-12-15 14:35:12.704 INFO  [730136] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 14:35:12.704 ERROR [730136] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3 --dbgnojump YES
    2017-12-15 14:35:12.704 FATAL [730136] [CRandomEngine::poststep@244] CRandomEngine::poststep rewind inhibited by option: --dbgnojump 
    flatExit: mrk:-# crfc:    8 df:1.10290554e-10 u_g4:0.714031577 u_ok:0.714031577 loc_g4:          OpBoundary loc_ok:          OpRayleigh  : lucf : 29    
    rayleigh_scatter_align p.direction (0 0 -1)
    rayleigh_scatter_align p.polarization (-0 1 -0)
    rayleigh_scatter_align.do u_OpRayleigh:0.714031577
     [  7]                                         OpRayleigh :     0.714031577 :    : 0.714031577 : 0.714031577 : 3 

    Process 51835 stopped





    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD 


    flatExit: mrk:-- crfc:    4 df:3.70178332e-11 u_g4:0.938713491 u_ok:0.938713491 loc_g4:OpBoundary_DiDiTransCoeff loc_ok:OpBoundary_DiDiTransCoeff  : lucf : 29    
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.938713491  reflect:1   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (  -37.8785    11.8230  -100.0000)
     [  3]                          OpBoundary_DiDiTransCoeff :     0.938713491 :    : 0.938713491 : 0.938713491 : 1 


    //                     opticks.ana.loc.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[1] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.938713  
    //                                                                       /_transmit : False 

    //                   opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[1] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:-- crfc:    5 df:3.44848594e-11 u_g4:0.753801465 u_ok:0.753801465 loc_g4:          OpBoundary loc_ok:          OpBoundary  : lucf : 29    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1
    propagate_to_boundary  u_OpBoundary:0.753801465 speed:299.79245
     [  4]                                         OpBoundary :     0.753801465 :    : 0.753801465 : 0.753801465 : 2 

    flatExit: mrk:-- crfc:    6 df:4.58282523e-10 u_g4:0.999846756 u_ok:0.999846756 loc_g4:          OpRayleigh loc_ok:          OpRayleigh  : lucf : 29    
    propagate_to_boundary  u_OpRayleigh:0.999846756   scattering_length(s.material1.z):1000000 scattering_distance:153.25528
     [  5]                                         OpRayleigh :     0.999846756 :    : 0.999846756 : 0.999846756 : 1 

    flatExit: mrk:-- crfc:    7 df:3.11492943e-10 u_g4:0.438019574 u_ok:0.438019574 loc_g4:        OpAbsorption loc_ok:        OpAbsorption  : lucf : 29    
    propagate_to_boundary  u_OpAbsorption:0.438019574   absorption_length(s.material1.y):10000000 absorption_distance:8254917
     [  6]                                       OpAbsorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

    2017-12-15 14:29:48.568 INFO  [727965] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 14:29:48.568 ERROR [727965] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3 --dbgnojump NO
    flatExit: mrk:*# crfc:    8 df:0.039769888 u_g4:0.753801465 u_ok:0.714031577 loc_g4:          OpBoundary loc_ok:          OpRayleigh  : lucf : 29    
    rayleigh_scatter_align p.direction (0 0 -1)
    rayleigh_scatter_align p.polarization (-0 1 -0)
    rayleigh_scatter_align.do u_OpRayleigh:0.714031577
     [  7]                                         OpRayleigh :     0.714031577 :    : 0.714031577 : 0.714031577 : 3 


    * OpBoundary is 1st consumption of the step



       1230 : /tmp/blyth/opticks/ox_1230.log  
     [  0]                                         OpBoundary :   0.00111702492 :    : 0.001117025 : 0.001117025 : 3 
     [  1]                                         OpRayleigh :      0.50264734 :    : 0.502647340 : 0.502647340 : 1 
     [  2]                                       OpAbsorption :     0.601504147 :    : 0.601504147 : 0.601504147 : 1 
     [  3]                          OpBoundary_DiDiTransCoeff :     0.938713491 :    : 0.938713491 : 0.938713491 : 1 

     [  4]                                         OpBoundary :    *0.753801465* :    : 0.753801465 : 0.753801465 : 2 
     [  5]                                         OpRayleigh :     0.999846756 :    : 0.999846756 : 0.999846756 : 1 
     [  6]                                       OpAbsorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

     [  7]                                         OpRayleigh :    *0.714031577* :    : 0.714031577 : 0.714031577 : 3 
     [  8]                                         OpRayleigh :     0.330403954 :    : 0.330403954 : 0.330403954 : 1 
     [  9]                                         OpRayleigh :     0.570741653 :    : 0.570741653 : 0.570741653 : 1 
     [ 10]                                         OpRayleigh :     0.375908673 :    : 0.375908673 : 0.375908673 : 1 
     [ 11]                                         OpRayleigh :      0.78497833 :    : 0.784978330 : 0.784978330 : 1 

     [ 12]                                         OpBoundary :     0.892654359 :    : 0.892654359 : 0.892654359 : 6 
     [ 13]                                         OpRayleigh :     0.441063195 :    : 0.441063195 : 0.441063195 : 1 
     [ 14]                                       OpAbsorption :     0.773742437 :    : 0.773742437 : 0.773742437 : 1 
     [ 15]                          OpBoundary_DiDiTransCoeff :     0.556839108 :    : 0.556839108 : 0.556839108 : 1 




What could go wrong with the rewind ?
----------------------------------------

* hmm why not seeing the burnt flatExit calls


::

    196 double CRandomEngine::flat()
    197 {
    198     if(!m_internal) m_location = CurrentProcessName();
    199     assert( m_current_record_flat_count < m_curand_nv );
    200     m_flat =  _flat() ;
    201     m_current_record_flat_count++ ; 
    202     m_current_step_flat_count++ ; 
    203     return m_flat ;   // (*lldb*) flatExit
    204 }   


    228 void CRandomEngine::poststep()
    229 {
    230     if(m_ctx._noZeroSteps > 0)
    231     {
    232         int backseq = -m_current_step_flat_count ;
    233         bool dbgnojump = m_ok->isDbgNoJump() ;
    234 
    235         LOG(error) << "CRandomEngine::poststep"
    236                    << " _noZeroSteps " << m_ctx._noZeroSteps
    237                    << " backseq " << backseq
    238                    << " --dbgnojump " << ( dbgnojump ? "YES" : "NO" )
    239                    ;
    240 
    241         if( dbgnojump )
    242         {
    243             LOG(fatal) << "CRandomEngine::poststep rewind inhibited by option: --dbgnojump " ;
    244         }
    245         else
    246         {
    247             jump(backseq);
    248         }
    249     }
    250 
    251     m_current_step_flat_count = 0 ;
    252 
    253     if( m_locseq )
    254     {
    255         m_locseq->poststep();
    256         LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ;
    257     }
    258 }




Full unmasked run into tag 2 : To find some record_id of non-maligned photons that scatter
--------------------------------------------------------------------------------------------

For access to some non-maligned photons that scatter, do a full run into tag 2

::

    tboolean-;TBOOLEAN_TAG=2 tboolean-box --okg4 --align 
    tboolean-;TBOOLEAN_TAG=2 tboolean-box-ip


    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dum
    ab.dump      ab.dumpline  

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


::

    In [1]: ab.aselhis = "TO BT SC BT SA"

    In [2]: ab.a.where
    Out[2]: array([ 4608, 17968, 61921, 86722, 91760, 93259, 94773])

    In [3]: ab.b.where
    Out[3]: array([ 4608, 17968, 61921, 86722, 91760, 93259, 94773])

    In [4]: ab.dumpline(ab.a.where)
          0   4608 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          1  17968 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          2  61921 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          3  86722 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          4  91760 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          5  93259 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          6  94773 :                                     TO BT SC BT SA                                     TO BT SC BT SA 


::

    tboolean-;tboolean-box --okg4 --align --mask 4608 --pindex 0 --pindexlog -DD 






Try blanket inhibiting the jump --dbgnojump
-----------------------------------------------

::

    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 9041  --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 14510 --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 49786 --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog -DD --dbgnojump   


Switching off the rewind with --dbgnojump keeps the RNG seq aligned, but get different 
seqhis-tories.  Need procName alignment checking too.






Who gets ahead on consumption ?
----------------------------------

::

   LOOKS LIKE AN UN-NEEDED -3 REWIND CAUSES THE MIS-ALIGN, 

   HMM SOME ZERO STEPS DONT NEED REWIND ?

   PERHAPS A ZERO STEP FOLLOWING A STEP IN WHICH THE BOUNDARY PROCESS WINS SHOULD NOT REWIND ?
 



69653 
~~~~~~~

::

    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog -DD 



    curi:69653 
       69653 : /tmp/blyth/opticks/ox_69653.log  
     [  0]                                      boundary_burn :    0.0819766819 :    : 0.081976682 : 0.081976682 : 3 
     [  1]                                         scattering :     0.490069658 :    : 0.490069658 : 0.490069658 : 1 
     [  2]                                         absorption :     0.800361693 :    : 0.800361693 : 0.800361693 : 1 
     [  3]                                            reflect :      0.50900209 :    : 0.509002090 : 0.509002090 : 1 
     [  4]                                      boundary_burn :     0.793467045 :    : 0.793467045 : 0.793467045 : 2 
     [  5]                                         scattering :     0.999958992 :    : 0.999958992 : 0.999958992 : 1 
     [  6]                                         absorption :     0.475769788 :    : 0.475769788 : 0.475769788 : 1 
     [  7]                                               rsa0 :     0.416864127 :    : 0.416864127 : 0.416864127 : 3 
     [  8]                                               rsa1 :     0.186498553 :    : 0.186498553 : 0.186498553 : 1 
     [  9]                                               rsa2 :     0.985090375 :    : 0.985090375 : 0.985090375 : 1 
     [ 10]                                               rsa3 :    0.0522525758 :    : 0.052252576 : 0.052252576 : 1 
     [ 11]                                               rsa4 :     0.308176816 :    : 0.308176816 : 0.308176816 : 1 
     [ 12]                                      boundary_burn :     0.471794218 :    : 0.471794218 : 0.471794218 : 6 
     [ 13]                                         scattering :     0.792557418 :    : 0.792557418 : 0.792557418 : 1 
     [ 14]                                         absorption :      0.47266078 :    : 0.472660780 : 0.472660780 : 1 
     [ 15]                                            reflect :    *0.160018712* :    : 0.160018712 : 0.160018712 : 1 
     [ 16]                                      boundary_burn :     0.539000034 :    : 0.539000034 : 0.539000034 : 2 
     [ 17]                                         scattering :     0.493351549 :    : 0.493351549 : 0.493351549 : 1 
     [ 18]                                         absorption :    *0.831078768* :    : 0.831078768 : 0.831078768 : 1 
     [ 19]                                            reflect :     0.995906353 :    : 0.995906353 : 0.995906353 : 1 
     [ 20]                                      boundary_burn :     0.828557372 :    : 0.828557372 : 0.828557372 : 2 
     [ 21]                                         scattering :     0.159997851 :    : 0.159997851 : 0.159997851 : 1 





     [ 13]                                         scattering :     0.792557418 :    : 0.792557418 : 0.792557418 : 1 

    flatExit: mrk:   crfc:   15 df:4.69970729e-11 flat:0.47266078  ufval:0.47266078 :        OpAbsorption; : lufc : 29    
    propagate_to_boundary  u_absorption:0.47266078   absorption_length(s.material1.y):1000000 absorption_distance:749377.312
     [ 14]                                         absorption :      0.47266078 :    : 0.472660780 : 0.472660780 : 1 


    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = TotalInternalReflection 
    flatExit: mrk:   crfc:   16 df:2.82180779e-10 flat:*0.160018712*  ufval:0.160018712 :          OpBoundary; : lufc : 29    
    propagate_at_boundary  u_reflect:    0.160018712  reflect:1   TransCoeff:   0.00000  c2c2:   -1.2761 tir:1  pos (  133.7670    10.0854  -100.0000)
     [ 15]                                            reflect :     0.160018712 :    : 0.160018712 : 0.160018712 : 1 

    flatExit: mrk:   crfc:   17 df:3.32275429e-10 flat:0.539000034  ufval:0.539000034 :          OpRayleigh; : lufc : 29    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3
    propagate_to_boundary  u_boundary_burn:0.539000034 speed:165.028061
     [ 16]                                      boundary_burn :     0.539000034 :    : 0.539000034 : 0.539000034 : 2 

    flatExit: mrk:   crfc:   18 df:8.98590091e-11 flat:0.493351549  ufval:0.493351549 :        OpAbsorption; : lufc : 29    
    propagate_to_boundary  u_scattering:0.493351549   scattering_length(s.material1.z):1000000 scattering_distance:706533.25
     [ 17]                                         scattering :     0.493351549 :    : 0.493351549 : 0.493351549 : 1 

    2017-12-15 11:21:33.840 INFO  [650846] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 11:21:33.840 ERROR [650846] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3


    flatExit: mrk:** crfc:   19 df:0.671060056 flat:0.160018712  ufval:0.831078768 :          OpBoundary; : lufc : 29    
    propagate_to_boundary  u_absorption:0.831078768   absorption_length(s.material1.y):1000000 absorption_distance:185030.703
     [ 18]                                         absorption :     0.831078768 :    : 0.831078768 : 0.831078768 : 1 

    Process 27386 stopped
    * thread #1: tid = 0x9ee5e, 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x00000001100ca580) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x00000001100ca580) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }
       208  
       209  




77962
~~~~~~~~

::

    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog -DD   


       77962 : /tmp/blyth/opticks/ox_77962.log  
     [  0]                                      boundary_burn :     0.587307692 :    : 0.587307692 : 0.587307692 : 3 
     [  1]                                         scattering :     0.367523879 :    : 0.367523879 : 0.367523879 : 1 
     [  2]                                         absorption :     0.368657529 :    : 0.368657529 : 0.368657529 : 1 
     [  3]                                            reflect :     0.883359611 :    : 0.883359611 : 0.883359611 : 1 
     [  4]                                      boundary_burn :     0.716171503 :    : 0.716171503 : 0.716171503 : 2 
     [  5]                                         scattering :    0.0115878591 :    : 0.011587859 : 0.011587859 : 1 
     [  6]                                         absorption :     0.265672505 :    : 0.265672505 : 0.265672505 : 1 
     [  7]                                            reflect :     0.959501982 :    : 0.959501982 : 0.959501982 : 1 
     [  8]                                      boundary_burn :    *0.974827707* :    : 0.974827707 : 0.974827707 : 2 
     [  9]                                         scattering :     0.999853075 :    : 0.999853075 : 0.999853075 : 1 
     [ 10]                                         absorption :     0.882926166 :    : 0.882926166 : 0.882926166 : 1 
     [ 11]                                               rsa0 :    *0.0676458701* :    : 0.067645870 : 0.067645870 : 3 
     [ 12]                                               rsa1 :     0.712023914 :    : 0.712023914 : 0.712023914 : 1 
     [ 13]                                               rsa2 :     0.388658017 :    : 0.388658017 : 0.388658017 : 1 
     [ 14]                                               rsa3 :     0.792805254 :    : 0.792805254 : 0.792805254 : 1 



    flatExit: mrk:   crfc:    8 df:2.64770539e-10 flat:0.959501982  ufval:0.959501982 :                      : lufc : 34    
    propagate_at_boundary  u_reflect:    0.959501982  reflect:1   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (  -29.0273    37.6855   100.0000)
     [  7]                                            reflect :     0.959501982 :    : 0.959501982 : 0.959501982 : 1 


    //                    opticks.ana.loc.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[25] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.959502  
    //                                                                       /_transmit : False 

    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:   crfc:    9 df:1.86187732e-10 flat:*0.974827707*  ufval:0.974827707 :          OpBoundary; : lufc : 34    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2
    propagate_to_boundary  u_boundary_burn:0.974827707 speed:165.028061
     [  8]                                      boundary_burn :     0.974827707 :    : 0.974827707 : 0.974827707 : 2 

    flatExit: mrk:   crfc:   10 df:4.49371318e-10 flat:0.999853075  ufval:0.999853075 :          OpRayleigh; : lufc : 34    
    propagate_to_boundary  u_scattering:0.999853075   scattering_length(s.material1.z):1000000 scattering_distance:146.936249
     [  9]                                         scattering :     0.999853075 :    : 0.999853075 : 0.999853075 : 1 

    flatExit: mrk:   crfc:   11 df:5.75867132e-11 flat:0.882926166  ufval:0.882926166 :        OpAbsorption; : lufc : 34    
    propagate_to_boundary  u_absorption:0.882926166   absorption_length(s.material1.y):1000000 absorption_distance:124513.695
     [ 10]                                         absorption :     0.882926166 :    : 0.882926166 : 0.882926166 : 1 

    2017-12-15 11:16:26.480 INFO  [649101] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 11:16:26.480 ERROR [649101] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3


    flatExit: mrk:** crfc:   12 df:0.907181837 flat:0.974827707  ufval:*0.0676458701* :          OpBoundary; : lufc : 34    
    rayleigh_scatter_align p.direction (-0 -0 -1)
    rayleigh_scatter_align p.polarization (0 -1 0)
    rayleigh_scatter_align.do u_rsa0:0.0676458701
     [ 11]                                               rsa0 :    0.0676458701 :    : 0.067645870 : 0.067645870 : 3 

    Process 27097 stopped
    * thread #1: tid = 0x9e78d, 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x000000010f602e80) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x000000010f602e80) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }
       208  
       209  




::


    tboolean-;tboolean-box --okg4 --align --mask 9041 --pindex 0 --pindexlog -DD 


    .[ 21]                                      boundary_burn :    *0.885444343*:    : 0.885444343 : 0.885444343 : 2 
     [ 22]                                         scattering :     0.554676592 :    : 0.554676592 : 0.554676592 : 1 
     [ 23]                                         absorption :     0.302562296 :    : 0.302562296 : 0.302562296 : 1  still together
     [ 24]                                            reflect :    *0.530730784* :    : 0.530730784 : 0.530730784 : 1 
     [ 25]                                      boundary_burn :      0.68599081 :    : 0.685990810 : 0.685990810 : 2 
     [ 26]                                         scattering :     0.601776481 :    : 0.601776481 : 0.601776481 : 1 
     [ 27]                                         absorption :     0.215921149 :    : 0.215921149 : 0.215921149 : 1 


     [ 20]                                            reflect :     0.921632886 :    : 0.921632886 : 0.921632886 : 1 


    //                    opticks.ana.loc.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[25] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                      /TransCoeff :  0.901669  
    //                                                                              /_u :  0.921633  
    //                                                                       /_transmit : False 

    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:   crfc:   22 df:9.0057406e-11 flat:*0.885444343*  ufval:0.885444343 :          OpBoundary; : lufc : 42    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3
    propagate_to_boundary  u_boundary_burn:0.885444343 speed:165.028061
     [ 21]                                      boundary_burn :     0.885444343 :    : 0.885444343 : 0.885444343 : 2 

    flatExit: mrk:   crfc:   23 df:3.50006135e-10 flat:0.554676592  ufval:0.554676592 :          OpRayleigh; : lufc : 42    
    propagate_to_boundary  u_scattering:0.554676592   scattering_length(s.material1.z):1000000 scattering_distance:589370.062
     [ 22]                                         scattering :     0.554676592 :    : 0.554676592 : 0.554676592 : 1 

    flatExit: mrk:   crfc:   24 df:3.90533439e-10 flat:0.302562296  ufval:0.302562296 :        OpAbsorption; : lufc : 42    
    propagate_to_boundary  u_absorption:0.302562296   absorption_length(s.material1.y):1000000 absorption_distance:1195468.12
     [ 23]                                         absorption :     0.302562296 :    : 0.302562296 : 0.302562296 : 1 

    2017-12-15 10:46:01.548 INFO  [639881] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 10:46:01.548 ERROR [639881] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3

               LOOKS LIKE AN UN-NEEDED -3 REWIND CAUSES THE MIS-ALIGN, 

               HMM SOME ZERO STEPS DONT NEED REWIND ?

               PERHAPS A ZERO STEP FOLLOWING A STEP IN WHICH THE BOUNDARY PROCESS WINS SHOULD NOT REWIND ?
               

    flatExit: mrk:** crfc:   25 df:0.354713559 flat:*0.885444343*  ufval:0.530730784 :          OpBoundary; : lufc : 42    
    propagate_at_boundary  u_reflect:    0.530730784  reflect:1   TransCoeff:   0.00000  c2c2:   -1.4179 tir:1  pos (   54.0247    85.2057  -100.0000)
     [ 24]                                            reflect :     0.530730784 :    : 0.530730784 : 0.530730784 : 1 

    Process 25885 stopped
    * thread #1: tid = 0x9c389, 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x0000000110856110) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x0000000110856110) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }





Debugging Idea
----------------

* common logging format for both simulations, so can just diff it 


Auto-interleave ?
-------------------

Redirect OptiX/CUDA logging to file ?
---------------------------------------

* https://stackoverflow.com/questions/21238303/redirecting-cuda-printf-to-a-c-stream

::

    simon:opticks blyth$ opticks-find rdbuf
    ./openmeshrap/MTool.cc:         cout_redirect out_(coutbuf.rdbuf());
    ./openmeshrap/MTool.cc:         cerr_redirect err_(cerrbuf.rdbuf()); 
    ./boostrap/BDirect.hh:        : old( std::cout.rdbuf( new_buffer ) ) 
    ./boostrap/BDirect.hh:        std::cout.rdbuf( old );
    ./boostrap/BDirect.hh:        : old( std::cerr.rdbuf( new_buffer ) ) 
    ./boostrap/BDirect.hh:        std::cerr.rdbuf( old );
    simon:opticks blyth$ 





First look at the 6 maligned
--------------------------------


::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


::

    In [20]: ab.dumpline(range(1220,1240))
          0   1220 :                                        TO BT BT SA                                        TO BT BT SA 
          1   1221 :                                        TO BT BT SA                                        TO BT BT SA 
          2   1222 :                                        TO BT BT SA                                        TO BT BT SA 
          3   1223 :                                        TO BT BT SA                                        TO BT BT SA 
          4   1224 :                                        TO BT BT SA                                        TO BT BT SA 
          5   1225 :                                        TO BT BT SA                                        TO BT BT SA 
          6   1226 :                                        TO BT BT SA                                        TO BT BT SA 
          7   1227 :                                        TO BT BT SA                                        TO BT BT SA 
          8   1228 :                                        TO BT BT SA                                        TO BT BT SA 
          9   1229 :                                        TO BT BT SA                                        TO BT BT SA 
         10   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
         11   1231 :                                        TO BT BT SA                                        TO BT BT SA 
         12   1232 :                                        TO BT BT SA                                        TO BT BT SA 
         13   1233 :                                        TO BT BT SA                                        TO BT BT SA 
         14   1234 :                                        TO BT BT SA                                        TO BT BT SA 
         15   1235 :                                        TO BT BT SA                                        TO BT BT SA 
         16   1236 :                                        TO BT BT SA                                        TO BT BT SA 
         17   1237 :                                        TO BT BT SA                                        TO BT BT SA 
         18   1238 :                                        TO BT BT SA                                        TO BT BT SA 
         19   1239 :                                           TO BR SA                                           TO BR SA 




1230 : could be reflectivity edge

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD   




::

    In [9]: ab.recline([1230,1230])
    Out[9]: '   1230   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA '


    In [18]: a.rpolw_(slice(0,8))[1230]
    Out[18]: 
    A()sliced
    A([    [ 0.    , -1.    ,  0.    , -0.1575],    TO
           [ 0.    ,  1.    ,  0.    , -0.1575],    BR
           [-0.1969, -0.9528, -0.2283, -0.1575],    SC
           [-0.685 , -0.7165,  0.1417, -0.1575],    BT
           [-0.685 ,  0.7165, -0.1417, -0.1575],    BR
           [-0.1732,  0.9528,  0.252 , -0.1575],
           [-0.1732,  0.9528,  0.252 , -0.1575],
           [-1.    , -1.    , -1.    , -1.    ]], dtype=float32)

    In [19]: b.rpolw_(slice(0,8))[1230]
    Out[19]: 
    A()sliced
    A([    [ 0.    , -1.    ,  0.    , -0.1575],   TO
           [ 0.    ,  1.    ,  0.    , -0.1575],   BR
           [-0.1969, -0.9528, -0.2283, -0.1575],   SC
           [-0.685 , -0.7165,  0.1417, -0.1575],   BT
           [-0.685 ,  0.7165, -0.1417, -0.1575],   BR
           [-0.315 ,  0.9449, -0.0551, -0.1575],
           [-0.3307,  0.937 , -0.1024, -0.1575],
           [-0.3307,  0.937 , -0.1024, -0.1575]], dtype=float32)





Maligned Six
---------------

::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 



Manually interleaving RNG consumption logging for 1230.

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD    



    2017-12-12 19:03:34.161 INFO  [146287] [CInputPhotonSource::GeneratePrimaryVertex@163] CInputPhotonSource::GeneratePrimaryVertex n 1
    2017-12-12 19:03:34.161 ERROR [146287] [CRandomEngine::pretrack@258] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[0] :    1   1  :  0.00111702  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[1] :    2   2  :  0.502647  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[2] :    3   3  :  0.601504  :  OpAbsorption;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[3] :    4   4  :  0.938713  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[0] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.000   -0.000    1.000)  
    //                                                                     .NewMomentum :  (   0.000    0.000    0.000)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.938713  
    //                                                                       /_transmit : False 
    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[0] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (   0.000    0.000   -1.000)  
    //                                                                     .NewMomentum :  (   0.000    0.000   -1.000)  


    2017-12-12 19:03:35.820 ERROR [146287] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:  0.00111702492 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.5026473403   scattering_length(s.material1.z):        1000000 scattering_distance:    687866.4375 
    propagate_to_boundary  u_absorption:   0.6015041471   absorption_length(s.material1.y):       10000000 absorption_distance:      5083218.5 
    propagate_at_boundary  u_reflect:       0.93871  reflect:1   TransCoeff:   0.93847 






    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[4] :    5   1  :  0.753801  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[5] :    6   2  :  0.999847  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[6] :    7   3  :  0.43802  :  OpAbsorption;   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1 
    propagate_to_boundary  u_boundary_burn:    0.753801465 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.9998467565   scattering_length(s.material1.z):        1000000 scattering_distance:    153.2552795 
    propagate_to_boundary  u_absorption:   0.4380195737   absorption_length(s.material1.y):       10000000 absorption_distance:        8254917 



    2017-12-12 19:03:34.663 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.663 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[7] :    8   1  :  0.753801  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[8] :    9   2  :  0.999847  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[9] :   10   3  :  0.43802  :  OpAbsorption;   


    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[10] :   11   4  :  0.714032  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[11] :   12   5  :  0.330404  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[12] :   13   6  :  0.570742  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[13] :   14   7  :  0.375909  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[14] :   15   8  :  0.784978  :  OpRayleigh;   

    rayleigh_scatter_align p.direction (0 0 -1) 
    rayleigh_scatter_align p.polarization (-0 1 -0) 
    rayleigh_scatter_align.do u0:0.714032 u1:0.330404 u2:0.570742 u3:0.375909 u4:0.784978 
    rayleigh_scatter_align.do constant        (0.301043) 
    rayleigh_scatter_align.do newDirection    (0.632086 -0.301043 0.714032) 
    rayleigh_scatter_align.do newPolarization (-0.199541 -0.953611 -0.225411) 
    rayleigh_scatter_align.do doCosTheta -0.953611 doCosTheta2 0.909373   looping 0   


    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[15] :   16   1  :  0.892654  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[16] :   17   2  :  0.441063  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[17] :   18   3  :  0.773742  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[18] :   19   4  :  0.556839  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   


    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2 
    propagate_to_boundary  u_boundary_burn:   0.8926543593 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4410631955   scattering_length(s.material1.z):        1000000 scattering_distance:     818567.125 
    propagate_to_boundary  u_absorption:   0.7737424374   absorption_length(s.material1.y):       10000000 absorption_distance:     2565162.25 
    propagate_at_boundary  u_reflect:       0.55684  reflect:0   TransCoeff:   0.88430 


    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[1] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (   0.632   -0.301    0.714)  
    //                                                                     .NewMomentum :  (   0.000    0.000   -1.000)  
    //                                                                      /TransCoeff :  0.884304  
    //                                                                              /_u :  0.556839  
    //                                                                       /_transmit : True 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[1] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (   0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (   0.381   -0.181    0.907)  







    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[19] :   20   1  :  0.775349  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[20] :   21   2  :  0.752141  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[21] :   22   3  :  0.412002  :  OpAbsorption;   



    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3 
    propagate_to_boundary  u_boundary_burn:    0.775349319 speed:    165.0280609 
    propagate_to_boundary  u_scattering:   0.7521412373   scattering_length(s.material1.z):        1000000 scattering_distance:    284831.1562 
    propagate_to_boundary  u_absorption:   0.4120023847   absorption_length(s.material1.y):        1000000 absorption_distance:     886726.125 
    propagate_at_boundary  u_reflect:       0.28246  reflect:1   TransCoeff:   0.00000  c2c2:   -1.3552 tir:1  pos (  150.0000   -77.6576    24.3052)   
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ WHATS THIS ??? DOES TIR CONSUME DIFFERENT ?



    In [7]: a.rpost_(slice(0,8))[1230]
    Out[7]: 
    A()sliced
    A([    [ -37.8781,   11.8231, -449.8989,    0.2002],    TO 
           [ -37.8781,   11.8231,  -99.9944,    1.3672],    BR   0
           [ -37.8781,   11.8231, -253.2548,    1.8781],    SC   1
           [  97.7921,  -52.7844,  -99.9944,    2.5941],    BT   2

           [ 149.9984,  -77.6556,   24.307 ,    3.4248],    BR   3   (point before was TIR)

           [ 118.2039,  -92.7959,   99.9944,    3.9308],   *BT*      << OK/G4 BT/BR
           [-191.6203, -240.3581,  449.9952,    5.566 ],   *SA*
           [   0.    ,    0.    ,    0.    ,    0.    ]])


    In [8]: b.rpost_(slice(0,8))[1230]
    Out[8]: 
    A()sliced
    A([    [ -37.8781,   11.8231, -449.8989,    0.2002],   TO
           [ -37.8781,   11.8231,  -99.9944,    1.3672],   BR 
           [ -37.8781,   11.8231, -253.2548,    1.8781],   SC
           [  97.7921,  -52.7844,  -99.9944,    2.5941],   BT
           [ 149.9984,  -77.6556,   24.307 ,    3.4248],   BR
           [ 118.2039,  -92.7959,   99.9944,    3.9308],  *BR* 
           [  34.2032, -132.8074,  -99.9944,    5.2675],  *BT*
           [-275.6348, -280.3696, -449.9952,    6.9027]]) *SA* 







    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:4 
    propagate_to_boundary  u_boundary_burn:   0.4324976802 speed:    165.0280609 
    propagate_to_boundary  u_scattering:   0.9078488946   scattering_length(s.material1.z):        1000000 scattering_distance:    96677.32812 
    propagate_to_boundary  u_absorption:   0.9121392369   absorption_length(s.material1.y):        1000000 absorption_distance:      91962.625 
    propagate_at_boundary  u_reflect:       0.20181  reflect:0   TransCoeff:   0.88556  c2c2:    0.5098 tir:0  pos (  118.2061   -92.8001   100.0000)   
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:5 
    propagate_to_boundary  u_boundary_burn:   0.7953493595 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4842039943   scattering_length(s.material1.z):        1000000 scattering_distance:         725249 
    propagate_to_boundary  u_absorption:  0.09354860336   absorption_length(s.material1.y):       10000000 absorption_distance:       23692742 
    propagate_at_surface   u_surface:       0.7505 
    propagate_at_surface   u_surface_burn:       0.9462 
    2017-12-12 19:32:41.223 ERROR [157506] [OPropagator::launch@185] LAUNCH DONE




















    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[2] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181    0.907)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[22] :   23   1  :  0.282463  :  OpBoundary;    <<< off-by-1
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[23] :   24   2  :  0.432498  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[24] :   25   3  :  0.907849  :  OpAbsorption;   

    2017-12-12 19:03:34.795 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.795 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3

    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[25] :   26   1  :  0.282463  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[26] :   27   2  :  0.432498  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[27] :   28   3  :  0.907849  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[28] :   29   4  :  0.912139  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[2] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                      /TransCoeff :  0.885559  
    //                                                                              /_u :  0.912139  
    //                                                                       /_transmit : False 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[3] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181   -0.907)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[29] :   30   1  :  0.201809  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[30] :   31   2  :  0.795349  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[31] :   32   3  :  0.484204  :  OpAbsorption;   
    2017-12-12 19:03:34.855 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.855 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[32] :   33   1  :  0.201809  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[33] :   34   2  :  0.795349  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[34] :   35   3  :  0.484204  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[35] :   36   4  :  0.0935486  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[3] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                      /TransCoeff :  0.874921  
    //                                                                              /_u :  0.0935486  
    //                                                                       /_transmit : True 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[4] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.632   -0.301   -0.714)  
    //                                                                     .NewMomentum :  (  -0.632   -0.301   -0.714)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[36] :   37   1  :  0.750533  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[37] :   38   2  :  0.946246  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[38] :   39   3  :  0.357591  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[39] :   40   4  :  0.166174  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[40] :   41   5  :  0.628917  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1242   

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[5] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.632   -0.301   -0.714)  
    //                                                                     .NewMomentum :  (  -0.632   -0.301   -0.714)  
    2017-12-12 19:03:34.926 INFO  [146287] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1

