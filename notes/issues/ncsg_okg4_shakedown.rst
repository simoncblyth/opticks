ncsg_okg4_shakedown
======================

tboolean-torus
-----------------

::

    tboolean-;tboolean-torus --okg4 
    tboolean-;tboolean-torus --okg4 --load --vizg4


NEXT
------

* currently using test material GlassShottF2, move to material care about 
* do some purely positional checks : profiting from the identical input photons 
  maybe add MaxVacuum with FLT_MAX extreme absorption_length   scattering_length


FIXED : Check of G4 geometry via GDML export fails : incomplete bordersurf
--------------------------------------------------------------------------------

* :doc:`ncsg_ggeotest_ctestdetector_cannot_gdml_export`


SC/AB in Vacuum
------------------

::

    simon:opticks blyth$ op --mat OpaqueVacuum
    === op-cmdline-binary-match : finds 1st argument with associated binary : --mat
    ubin /usr/local/opticks/lib/GMaterialLibTest cfm --mat cmdline --mat OpaqueVacuum
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/GMaterialLibTest
    256 -rwxr-xr-x  1 blyth  staff  129400 Nov  1 14:45 /usr/local/opticks/lib/GMaterialLibTest
    proceeding.. : /usr/local/opticks/lib/GMaterialLibTest --mat OpaqueVacuum
      0 : /usr/local/opticks/lib/GMaterialLibTest
      1 : --mat
      2 : OpaqueVacuum
    option '--mat' is ambiguous and matches '--materialdbg', and '--materialprefix'
    2017-11-01 20:58:00.795 INFO  [2230869] [main@109]  ok 
    2017-11-01 20:58:00.799 INFO  [2230869] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-11-01 20:58:00.799 INFO  [2230869] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-11-01 20:58:00.799 INFO  [2230869] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-11-01 20:58:00.800 INFO  [2230869] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::dumpDomain@161] GPropertyLib::dumpDomain
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::dumpDomain@163]  low/high/step  low  60 high 820 step 20 dscale 0.00123984 dscale/low 2.0664e-05 dscale/high 1.512e-06
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::dumpDomain@172] GPropertyLib::dumpDomain GC::nanometer 1e-06 GC::h_Planck 4.13567e-12 GC::c_light (mm/ns ~299.792) 299.792 dscale 0.00123984
    2017-11-01 20:58:00.801 INFO  [2230869] [main@115]  after load 
    F2 ri : b0ad5d685c9b6bfb9cbcb3d68e3a3024 : 101 
    d     320.000   2500.000
    v       1.696      1.582
    2017-11-01 20:58:00.801 INFO  [2230869] [GMaterialLib::Summary@220] dump NumMaterials 39 NumFloat4 2
    2017-11-01 20:58:00.801 INFO  [2230869] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [OpaqueVacuum]
    2017-11-01 20:58:00.802 INFO  [2230869] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 39,2,39,4
                  domain    refractive_index   absorption_length   scattering_length     reemission_prob      group_velocity
                      60                   1               1e+06               1e+06                   0             299.792
                      80                   1               1e+06               1e+06                   0             299.792
                     100                   1               1e+06               1e+06                   0             299.792
                     120                   1               1e+06               1e+06                   0             299.792
                     140                   1               1e+06               1e+06                   0             299.792
                     160                   1               1e+06               1e+06                   0             299.792
                     180                   1               1e+06               1e+06                   0             299.792
                     200                   1               1e+06               1e+06                   0             299.792





MainH2OHale sphere-in-box : good agreement
---------------------------------------------

::

    [2017-11-02 15:27:32,091] p47470 {/Users/blyth/opticks/ana/ab.py:133} INFO - AB.init_point DONE
    AB(1,torch,tboolean-sphere)  None 0 
    A tboolean-sphere/torch/  1 :  20171102-1527 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/1/fdom.npy 
    B tboolean-sphere/torch/ -1 :  20171102-1527 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///MainH2OHale
    .                seqhis_ana  1:tboolean-sphere   -1:tboolean-sphere        c2        ab        ba 
    .                             600000    600000        17.45/10 =  1.75  (pval:0.065 prob:0.935)  
    0000     343380    344103             0.76  TO BT BT SA
    0001     210643    210641             0.00  TO SA
    0002      26154     25966             0.68  TO BR SA
    0003      16090     15731             4.05  TO BT BR BT SA
    0004       2419      2278             4.23  TO BT BR BR BT SA
    0005        689       675             0.14  TO BT BR BR BR BT SA
    0006        265       270             0.05  TO BT BR BR BR BR BT SA
    0007        153       127             2.41  TO BT BR BR BR BR BR BT SA
    0008         80        78             0.03  TO BT BR BR BR BR BR BR BR BR
    0009         69        55             1.58  TO BT BR BR BR BR BR BR BT SA
    0010         37        55             3.52  TO BT BR BR BR BR BR BR BR BT
    0011          8         7             0.00  TO BT AB
    0012          8         5             0.00  TO SC SA
    0013          3         3             0.00  TO BT SC BT SA
    0014          0         2             0.00  TO AB
    0015          0         1             0.00  TO BT BT SC BT BT SA
    0016          0         1             0.00  TO BT BT SC BR SA
    0017          1         1             0.00  TO BT SC BR BT SA
    0018          0         1             0.00  TO SC BT BR BT SA
    0019          1         0             0.00  TO SC BT BT SA
    .                             600000    600000        17.45/10 =  1.75  (pval:0.065 prob:0.935)  


Why MainH2OHale so good ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`material_review`




Pyrex sphere-in-box very messed up : must be material conversion issue ?
---------------------------------------------------------------------------

finely binned prop values of --mat are not being dumped with --cmat ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    op --cmat Pyrex
    op --mat Pyrex



Tangent : NCSG emitonly metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some CSG metadata that marks geometry as emitter lightsource only, which 
can skip from geometry point of view, would allow convenient planting of
emitters of any shape/position.


Converted G4 Pyrex absorbing immediately 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* debug attempt failed to materialize anything within G4 code
* so try rebuild G4 with Debug config (it was using RelWithDebInfo)  


Debug Ideas
~~~~~~~~~~~~~~

* put photon source inside pyrex : so all photons act the same for ease of debug


::

    simon:opticks blyth$ tboolean-;tboolean-sphere-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-sphere --tag 1
    ok.smry 1 
    [2017-11-02 15:22:11,429] p46943 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-sphere c2max 2.0 ipython False 
    [2017-11-02 15:22:11,429] p46943 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-02 15:22:11,463] p46943 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -11.000  11.000 : tot 600000 over 278 0.000  under 265 0.000 : mi    -11.000 mx     11.000  
    [2017-11-02 15:22:11,472] p46943 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -11.000  11.000 : tot 600000 over 262 0.000  under 286 0.000 : mi    -11.000 mx     11.000  
    [2017-11-02 15:22:11,479] p46943 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  z : -11.000  11.000 : tot 600000 over 282 0.000  under 285 0.000 : mi    -11.000 mx     11.000  
    [2017-11-02 15:22:12,223] p46943 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-02 15:22:12,226] p46943 {/Users/blyth/opticks/ana/ab.py:131} INFO - AB.init_point START
    [2017-11-02 15:22:12,228] p46943 {/Users/blyth/opticks/ana/ab.py:133} INFO - AB.init_point DONE
    AB(1,torch,tboolean-sphere)  None 0 
    A tboolean-sphere/torch/  1 :  20171102-1521 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/1/fdom.npy 
    B tboolean-sphere/torch/ -1 :  20171102-1521 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///Pyrex
    .                seqhis_ana  1:tboolean-sphere   -1:tboolean-sphere        c2        ab        ba 
    .                             600000    600000    691731.17/13 = 53210.09  (pval:0.000 prob:1.000)  
    0000       5217    356055        340705.35  TO BT AB
    0001     326726         0        326726.00  TO BT BT SA
    0002     210643    210643             0.00  TO SA
    0003      33063     33297             0.83  TO BR SA
    0004      19223         0         19223.00  TO BT BR BT SA
    0005       3108         0          3108.00  TO BT BR BR BT SA
    0006        839         0           839.00  TO BT BR BR BR BT SA
    0007        356         0           356.00  TO BT BR AB
    0008        308         0           308.00  TO BT BR BR BR BR BT SA
    0009        183         0           183.00  TO BT BR BR BR BR BR BT SA
    0010         94         0            94.00  TO BT BR BR BR BR BR BR BT SA
    0011         92         0            92.00  TO BT BR BR BR BR BR BR BR BR
    0012         56         0            56.00  TO BT BR BR AB
    0013         40         0            40.00  TO BT BR BR BR BR BR BR BR BT
    0014         18         0             0.00  TO BT BR BR BR AB
    0015         10         0             0.00  TO BT BR BR BR BR AB
    0016          8         5             0.00  TO SC SA
    0017          5         0             0.00  TO BT BR BR BR BR BR AB
    0018          4         0             0.00  TO BT BR BR BR BR BR BR BR AB
    0019          4         0             0.00  TO BT SC BT SA
    .                             600000    600000    691731.17/13 = 53210.09  (pval:0.000 prob:1.000)  


sphere-in-box OKish
----------------------

::

    simon:opticks blyth$ tboolean-;tboolean-sphere-p


    [2017-11-02 15:11:41,610] p46299 {/Users/blyth/opticks/ana/ab.py:133} INFO - AB.init_point DONE
    AB(1,torch,tboolean-sphere)  None 0 
    A tboolean-sphere/torch/  1 :  20171102-1436 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/1/fdom.npy 
    B tboolean-sphere/torch/ -1 :  20171102-1436 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    .                seqhis_ana  1:tboolean-sphere   -1:tboolean-sphere        c2        ab        ba 
    .                             600000    600000       194.16/10 = 19.42  (pval:0.000 prob:1.000)  
    0000     312582    317268            34.86  TO BT BT SA
    0001     210643    210641             0.00  TO SA
    0002      44427     41861            76.31  TO BR SA            <<<< Opticks relecting more
    0003      25335     23872            43.50  TO BT BR BT SA
    0004       4641      4156            26.74  TO BT BR BR BT SA
    0005       1276      1135             8.25  TO BT BR BR BR BT SA
    0006        473       497             0.59  TO BT BR BR BR BR BT SA
    0007        246       206             3.54  TO BT BR BR BR BR BR BT SA
    0008        153       149             0.05  TO BT BR BR BR BR BR BR BR BR
    0009        129       123             0.14  TO BT BR BR BR BR BR BR BT SA
    0010         71        66             0.18  TO BT BR BR BR BR BR BR BR BT
    0011          8         8             0.00  TO BT AB
    0012          8         8             0.00  TO SC SA
    0013          1         4             0.00  TO SC BT BT SA
    0014          3         1             0.00  TO BT SC BT SA
    0015          1         2             0.00  TO BT SC BR BR BR BR BR BR BR
    0016          1         1             0.00  TO BT SC BR BT SA
    0017          1         0             0.00  TO BT BR AB
    0018          1         0             0.00  TO BT BR SC BR BR BR BR BR BR
    0019          0         1             0.00  TO BT BR BR SC BR BR BR BR BR
    .                             600000    600000       194.16/10 = 19.42  (pval:0.000 prob:1.000)  


::

    simon:opticks blyth$ tboolean-;tboolean-sphere-a
    2017-11-02 14:37:55.348 INFO  [2328424] [Opticks::dumpArgs@806] Opticks::configure argc 10
      0 : OpticksEventCompareTest
      1 : --torch
      2 : --tag
      3 : 1
      4 : --cat
      5 : tboolean-sphere
      6 : --dbgnode
      7 : 0
      8 : --dbgseqhis
      9 : 0x86d
    ...

    2017-11-02 14:37:59.018 INFO  [2328424] [*OpticksEventStat::CreateRecordsNPY@33] OpticksEventStat::CreateRecordsNPY  shape 600000,10,2,4
    2017-11-02 14:37:59.047 INFO  [2328424] [OpticksEventCompare::dump@20] cf(evt,g4evt)
    2017-11-02 14:37:59.047 INFO  [2328424] [OpticksEventStat::dump@86] A evt Evt /tmp/blyth/opticks/evt/tboolean-sphere/torch/1 20171102_143639 /usr/local/opticks/lib/OKG4Test totmin 2
     seqhis             8ccd                 TO BT BT SA                                      tot 312582
     seqhis               8d                 TO SA                                            tot 210643
     seqhis              8bd                 TO BR SA                                         tot  44427   <<<< opticks reflecting more
     seqhis            8cbcd                 TO BT BR BT SA                                   tot  25335
     seqhis           8cbbcd                 TO BT BR BR BT SA                                tot   4641
     seqhis          8cbbbcd                 TO BT BR BR BR BT SA                             tot   1276
     seqhis         8cbbbbcd                 TO BT BR BR BR BR BT SA                          tot    473
     seqhis        8cbbbbbcd                 TO BT BR BR BR BR BR BT SA                       tot    246
     seqhis       bbbbbbbbcd                 TO BT BR BR BR BR BR BR BR BR                    tot    153
     seqhis       8cbbbbbbcd                 TO BT BR BR BR BR BR BR BT SA                    tot    129
     seqhis       cbbbbbbbcd                 TO BT BR BR BR BR BR BR BR BT                    tot     71
     seqhis              4cd                 TO BT AB                                         tot      8
     seqhis              86d                 TO SC SA                                         tot      8
     seqhis            8c6cd                 TO BT SC BT SA                                   tot      3
    2017-11-02 14:37:59.047 INFO  [2328424] [OpticksEventStat::dump@86] B evt Evt /tmp/blyth/opticks/evt/tboolean-sphere/torch/-1 20171102_143639 /usr/local/opticks/lib/OKG4Test totmin 2
     seqhis             8ccd                 TO BT BT SA                                      tot 317268
     seqhis               8d                 TO SA                                            tot 210641
     seqhis              8bd                 TO BR SA                                         tot  41861
     seqhis            8cbcd                 TO BT BR BT SA                                   tot  23872
     seqhis           8cbbcd                 TO BT BR BR BT SA                                tot   4156
     seqhis          8cbbbcd                 TO BT BR BR BR BT SA                             tot   1135
     seqhis         8cbbbbcd                 TO BT BR BR BR BR BT SA                          tot    497
     seqhis        8cbbbbbcd                 TO BT BR BR BR BR BR BT SA                       tot    206
     seqhis       bbbbbbbbcd                 TO BT BR BR BR BR BR BR BR BR                    tot    149
     seqhis       8cbbbbbbcd                 TO BT BR BR BR BR BR BR BT SA                    tot    123
     seqhis       cbbbbbbbcd                 TO BT BR BR BR BR BR BR BR BT                    tot     66
     seqhis              4cd                 TO BT AB                                         tot      8
     seqhis              86d                 TO SC SA                                         tot      8
     seqhis            8cc6d                 TO SC BT BT SA                                   tot      4
     seqhis       bbbbbbb6cd                 TO BT SC BR BR BR BR BR BR BR                    tot      2
    simon:opticks blyth$ 



sphere-in-sphere : G4 barfing loadsa warnings : "Logic error: snxt = kInfinity"
------------------------------------------------------------------------------------

* INTERIM CONCLUSION : **G4 doesnt like normal incidence onto a sphere** ? 

* no such issue from box-in-box or sphere-in-box ?

* perhaps edge problem : are starting the photon on the outer sphere (edge of the world) 

  * NOPE : adding NEmitConfig.posdelta to nudge start position along 
    its direction (the normal) doesnt avoid the issue

* for easy debug use spheres of 100mm and 10mm


::

    tboolean-;tboolean-sphere --okg4
    ...

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------


    -----------------------------------------------------------
        *** Dump for solid - sphere ***
        ===================================================
     Solid type: G4Sphere
     Parameters: 
        inner radius: 0 mm 
        outer radius: 10 mm 
        starting phi of segment  : 0 degrees 
        delta phi of segment     : 360 degrees 
        starting theta of segment: 0 degrees 
        delta theta of segment   : 180 degrees 
    -----------------------------------------------------------

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : GeomSolids1002
          issued by : G4Sphere::DistanceToOut(p,v,..)
    Logic error: snxt = kInfinity  ???
    Position:

    p.x() = -0.05812894200256247 mm
    p.y() = 0.1384359192676456 mm
    p.z() = -9.998881795334469 mm

    Rp = 10.00000903173157 mm

    Direction:

    v.x() = 0.005812884243438132
    v.y() = -0.01384358278837826
    v.z() = 0.9998872764428766

    Proposed distance :

    snxt = 9e+99 mm

    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------




FIXED : tboolean-sphere : sphere in sphere bizarre lissajoux like pattern
-----------------------------------------------------------------------------

Fixed by saving source photons with the OpticksEvent, 
observing incomplete coverage with so.py 
and fixing bug in nsphere::par_posnrm_model
 
::

    ipython -i $(which so.py) -- --det tboolean-sphere --tag 1 --src torch 

    In [4]: v = so[:,0,:3]

    In [8]: from opticks.ana.nbase import vnorm

    In [9]: vnorm(v)
    Out[9]: 
    A()sliced
    A([ 400.,  400.,  400., ...,  400.,  400.,  400.], dtype=float32)


    In [12]: v[:,0].min()
    Out[12]: 
    A()sliced
    A(-400.0, dtype=float32)

    In [13]: v[:,0].max()    ## this should be +400 
    Out[13]: 
    A()sliced
    A(108.86621856689453, dtype=float32)


tboolean-box also shows BR discrep
-------------------------------------------

* hmm are the material props being translated correctly ?


::

    tboolean-box --okg4

    simon:opticksgeo blyth$ tboolean-;tboolean-box-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    ok.smry 1 
    [2017-11-01 20:50:38,288] p20501 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2017-11-01 20:50:38,288] p20501 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:50:38,331] p20501 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -600.000 600.000 : tot 600000 over 13 0.000  under 22 0.000 : mi   -600.000 mx    600.000  
    [2017-11-01 20:50:38,339] p20501 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -600.000 600.000 : tot 600000 over 6 0.000  under 8 0.000 : mi   -600.000 mx    600.000  
    [2017-11-01 20:50:38,349] p20501 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  z : -600.000 600.000 : tot 600000 over 8 0.000  under 5 0.000 : mi   -600.000 mx    600.000  
    [2017-11-01 20:50:39,004] p20501 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:50:39,008] p20501 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:50:39,010] p20501 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171101-2049 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy 
    B tboolean-box/torch/ -1 :  20171101-2049 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000        16.79/6 =  2.80  (pval:0.010 prob:0.990)  
    0000     570058    570041             0.00  TO SA
    0001      25702     25962             1.31  TO BT BT SA
    0002       1799      1594            12.39  TO BR SA
    0003       1536      1498             0.48  TO BT BR BT SA
    0004        694       698             0.01  TO SC SA
    0005         97        82             1.26  TO BT BR BR BT SA
    0006         56        69             1.35  TO AB
    0007         15         8             0.00  TO BT BT SC SA
    0008         11        11             0.00  TO SC BT BT SA
    0009         10         3             0.00  TO BT BR BR BR BT SA
    0010          6         7             0.00  TO BT AB
    0011          6         5             0.00  TO SC BT BR BT SA
    0012          2         5             0.00  TO BT SC BR BR BR BR BR BR BR
    0013          1         4             0.00  TO SC BR SA
    0014          3         3             0.00  TO BT SC BR BT SA
    0015          1         3             0.00  TO SC BT BR BR BT SA
    0016          0         3             0.00  TO BT SC BT SA
    0017          0         1             0.00  TO BT BR BT SC SA
    0018          0         1             0.00  TO SC BT BR BR BR BR BT SA
    0019          1         0             0.00  TO BT BR SC BR BR BR BT SA
    .                             600000    600000        16.79/6 =  2.80  (pval:0.010 prob:0.990)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 



Avoid the touching container : see BR discrep
------------------------------------------------

::

    simon:opticksgeo blyth$ tboolean-;tboolean-torus-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-torus --tag 1
    ok.smry 1 
    [2017-11-01 20:40:38,373] p20189 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-torus c2max 2.0 ipython False 
    [2017-11-01 20:40:38,373] p20189 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:40:38,441] p20189 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -150.500 150.500 : tot 600000 over 105 0.000  under 86 0.000 : mi   -150.500 mx    150.500  
    [2017-11-01 20:40:38,449] p20189 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -150.500 150.500 : tot 600000 over 77 0.000  under 93 0.000 : mi   -150.500 mx    150.500  
    [2017-11-01 20:40:39,460] p20189 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:40:39,482] p20189 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:40:39,498] p20189 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-torus)  None 0 
    A tboolean-torus/torch/  1 :  20171101-2039 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/1/fdom.npy 
    B tboolean-torus/torch/ -1 :  20171101-2039 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 
    .                             600000    600000      1052.10/42 = 25.05  (pval:0.000 prob:1.000)  
    0000     196365    205447           205.28  TO BT BT SA
    0001     100590     96737            75.23  TO BT BR BT SA
    0002      94658     94651             0.00  TO SA
    0003      54961     52006            81.63  TO BR SA
    0004      42289     45580           123.26  TO BT BT BT BT SA
    0005      33255     29115           274.81  TO BT BR BR BR BR BR BR BR BR
    0006      16959     18197            43.60  TO BT BR BR BR BT SA
    0007      15456     14218            51.65  TO BT BR BR BR BR BT SA
    0008      10597     11409            29.96  TO BT BR BR BT SA
    0009      11331     10678            19.37  TO BT BR BR BR BR BR BT SA
    0010       6901      5817            92.39  TO BT BR BR BR BR BR BR BR BT
    0011       6804      6464             8.71  TO BT BR BR BR BR BR BR BT SA
    0012       3139      3022             2.22  TO BT BT BR SA
    0013       1852      1917             1.12  TO BT BT BT BR BT SA
    0014       1402      1516             4.45  TO BT BT BR BT BT SA
    0015        711       652             2.55  TO BT BT BT BR BT BT BT SA
    0016        470       454             0.28  TO BR BT BT SA
    0017        408       361             2.87  TO BT BR BR BT BT BT SA
    0018        292       260             1.86  TO BT BT BT BR BR BT SA
    0019        196       187             0.21  TO BT BT BR BR SA
    .                             600000    600000      1052.10/42 = 25.05  (pval:0.000 prob:1.000)  
    .                pflags_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 



with overtight (touching container) : crazy MI
------------------------------------------------

::

    simon:opticksgeo blyth$ tboolean-torus-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-torus --tag 1
    ok.smry 1 
    [2017-11-01 20:30:41,828] p19231 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-torus c2max 2.0 ipython False 
    [2017-11-01 20:30:41,828] p19231 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:30:41,900] p19231 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -150.000 150.000 : tot 600000 over 80 0.000  under 83 0.000 : mi   -150.000 mx    150.000  
    [2017-11-01 20:30:41,907] p19231 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -150.000 150.000 : tot 600000 over 88 0.000  under 76 0.000 : mi   -150.000 mx    150.000  
    [2017-11-01 20:30:43,012] p19231 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:30:43,104] p19231 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:30:43,125] p19231 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-torus)  None 0 
    A tboolean-torus/torch/  1 :  20171101-2028 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/1/fdom.npy 
    B tboolean-torus/torch/ -1 :  20171101-2028 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 
    .                             600000    600000     58933.95/53 = 1111.96  (pval:0.000 prob:1.000)  
    0000     151079    207121          8768.02  TO BT BT SA
    0001     101285     98084            51.39  TO BT BR BT SA
    0002      88847     88850             0.00  TO SA
    0003      54915     52564            51.43  TO BR SA
    0004      42258     46593           211.50  TO BT BT BT BT SA
    0005      39350         0         39350.00  TO BT MI
    0006      33754     29379           303.18  TO BT BR BR BR BR BR BR BR BR
    0007      17192     18450            44.40  TO BT BR BR BR BT SA
    0008      15683     14282            65.50  TO BT BR BR BR BR BT SA
    0009      10562     11662            54.45  TO BT BR BR BT SA
    0010      11270     10721            13.71  TO BT BR BR BR BR BR BT SA
    0011       8175         0          8175.00  TO MI
    0012       7183      5915           122.75  TO BT BR BR BR BR BR BR BR BT
    0013       6754      6707             0.16  TO BT BR BR BR BR BR BR BT SA
    0014       3201      3075             2.53  TO BT BT BR SA
    0015       1871      2019             5.63  TO BT BT BT BR BT SA
    0016       1378      1422             0.69  TO BT BT BR BT BT SA
    0017        683       633             1.90  TO BT BT BT BR BT BT BT SA
    0018        486       457             0.89  TO BR BT BT SA
    0019        462         0           462.00  TO BT BT BT SA
    .                             600000    600000     58933.95/53 = 1111.96  (pval:0.000 prob:1.000)  



poor chi2 : but wasting most of the stats
-------------------------------------------

::

    simon:opticksgeo blyth$ tboolean-;tboolean-torus-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-torus --tag 1
    ok.smry 1 
    [2017-11-01 20:21:41,719] p18277 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-torus c2max 2.0 ipython False 
    [2017-11-01 20:21:41,719] p18277 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-01 20:21:41,758] p18277 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  x : -400.000 400.000 : tot 600000 over 868 0.001  under 785 0.001 : mi   -400.000 mx    400.000  
    [2017-11-01 20:21:41,766] p18277 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  y : -400.000 400.000 : tot 600000 over 802 0.001  under 813 0.001 : mi   -400.000 mx    400.000  
    [2017-11-01 20:21:41,773] p18277 {/Users/blyth/opticks/ana/evt.py:392} WARNING -  z : -400.000 400.000 : tot 600000 over 1998 0.003  under 1944 0.003 : mi   -400.000 mx    400.000  
    [2017-11-01 20:21:42,467] p18277 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-01 20:21:42,477] p18277 {/Users/blyth/opticks/ana/ab.py:125} INFO - AB.init_point START
    [2017-11-01 20:21:42,485] p18277 {/Users/blyth/opticks/ana/ab.py:127} INFO - AB.init_point DONE
    AB(1,torch,tboolean-torus)  None 0 
    A tboolean-torus/torch/  1 :  20171101-2000 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/1/fdom.npy 
    B tboolean-torus/torch/ -1 :  20171101-2000 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/fdom.npy 
    .                seqhis_ana  1:tboolean-torus   -1:tboolean-torus        c2        ab        ba 
    .                             600000    600000        65.09/19 =  3.43  (pval:0.000 prob:1.000)  
    0000     562547    562537             0.00  TO SA
    0001      20117     20771            10.46  TO BT BT SA
    0002       5625      5365             6.15  TO BT BR BT SA
    0003       3780      3428            17.19  TO BR SA
    0004       2050      2168             3.30  TO BT BT BT BT SA
    0005       1577      1402            10.28  TO BT BR BR BR BR BR BR BR BR
    0006        768       858             4.98  TO BT BR BR BR BT SA
    0007        748       688             2.51  TO BT BR BR BR BR BT SA
    0008        593       601             0.05  TO BT BR BR BT SA
    0009        516       510             0.04  TO BT BR BR BR BR BR BT SA
    0010        458       472             0.21  TO SC SA
    0011        327       278             3.97  TO BT BR BR BR BR BR BR BR BT
    0012        289       311             0.81  TO BT BR BR BR BR BR BR BT SA
    0013        156       156             0.00  TO BT BT BR SA
    0014         88        87             0.01  TO BT BT BT BR BT SA
    0015         54        73             2.84  TO BT BT BR BT BT SA
    0016         62        58             0.13  TO BR BT BT SA
    0017         41        41             0.00  TO AB
    0018         26        35             1.33  TO BT BT BT BR BT BT BT SA
    0019         26        33             0.83  TO BT BR BR BT BT BT SA
    .                             600000    600000        65.09/19 =  3.43  (pval:0.000 prob:1.000)  



tboolean_torus with CPU side photons
---------------------------------------

Emitted input photons are exactly the same in both simulations, 
so should be able to get very close matching. After turn off things
scattering/absorption ? Perhaps use different flavors of vacuum to do this ? 



Difference in ox flags causes different np dumping::

    simon:ana blyth$ ox.py --det tboolean-torus  --tag 1 
    args: /Users/blyth/opticks/ana/ox.py --det tboolean-torus --tag 1
    [2017-11-01 18:21:31,501] p15395 {/Users/blyth/opticks/ana/ox.py:32} INFO - loaded ox /tmp/blyth/opticks/evt/tboolean-torus/torch/1/ox.npy 20171101-1515 shape (600000, 4, 4) 
    [[[-386.263  -310.873   400.        2.8685]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ -14.892  -262.1473  400.        2.8685]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ 333.2202 -201.3483  400.        2.8685]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.      380.    ]
      [   0.        0.        0.        0.    ]]

     ..., 
     [[-174.9729 -400.      253.6111    2.8685]
      [  -0.       -1.       -0.        1.    ]
      [   0.        0.       -1.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ 259.2407 -400.     -149.578     2.8685]
      [  -0.       -1.       -0.        1.    ]
      [   0.        0.       -1.      380.    ]
      [   0.        0.        0.        0.    ]]

     [[ -64.378  -400.     -129.1872    2.8685]
      [  -0.       -1.       -0.        1.    ]
      [   0.        0.       -1.      380.    ]
      [   0.        0.        0.        0.    ]]]


::

    simon:ana blyth$ ox.py --det tboolean-torus  --tag -1 
    args: /Users/blyth/opticks/ana/ox.py --det tboolean-torus --tag -1
    [2017-11-01 18:21:48,799] p15402 {/Users/blyth/opticks/ana/ox.py:32} INFO - loaded ox /tmp/blyth/opticks/evt/tboolean-torus/torch/-1/ox.npy 20171101-1515 shape (600000, 4, 4) 
    [[[ -3.8626e+02  -3.1087e+02   4.0000e+02   2.8685e+00]
      [ -0.0000e+00  -0.0000e+00   1.0000e+00   1.0000e+00]
      [  0.0000e+00  -1.0000e+00   0.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[ -1.4892e+01  -2.6215e+02   4.0000e+02   2.8685e+00]
      [ -0.0000e+00  -0.0000e+00   1.0000e+00   1.0000e+00]
      [  0.0000e+00  -1.0000e+00   0.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[  3.3322e+02  -2.0135e+02   4.0000e+02   2.8685e+00]
      [ -0.0000e+00  -0.0000e+00   1.0000e+00   1.0000e+00]
      [  0.0000e+00  -1.0000e+00   0.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     ..., 
     [[ -1.7497e+02  -4.0000e+02   2.5361e+02   2.8685e+00]
      [ -0.0000e+00  -1.0000e+00  -0.0000e+00   1.0000e+00]
      [  0.0000e+00   0.0000e+00  -1.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[  2.5924e+02  -4.0000e+02  -1.4958e+02   2.8685e+00]
      [ -0.0000e+00  -1.0000e+00  -0.0000e+00   1.0000e+00]
      [  0.0000e+00   0.0000e+00  -1.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]

     [[ -6.4378e+01  -4.0000e+02  -1.2919e+02   2.8685e+00]
      [ -0.0000e+00  -1.0000e+00  -0.0000e+00   1.0000e+00]
      [  0.0000e+00   0.0000e+00  -1.0000e+00   3.8000e+08]
      [  2.8026e-45   0.0000e+00   1.5400e-36   5.9191e-42]]]
    simon:ana blyth$ 


