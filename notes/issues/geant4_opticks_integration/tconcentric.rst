tconcentric
==============


setup
---------

Concentric spheres 3m 4m 5m  with default random radial torch

::

     52     local test_config=(
     53                  mode=BoxInBox
     54                  analytic=1
     55     
     56                  shape=sphere
     57                  boundary=Acrylic//RSOilSurface/MineralOil
     58                  parameters=0,0,0,5000
     59 
     60 
     61                  shape=sphere
     62                  boundary=MineralOil///Acrylic
     63                  parameters=0,0,0,$(( 4000 + 5 ))
     64 
     65                  shape=sphere
     66                  boundary=Acrylic///LiquidScintillator
     67                  parameters=0,0,0,$(( 4000 - 5 ))
     68 
     69 
     70                  shape=sphere
     71                  boundary=LiquidScintillator///Acrylic
     72                  parameters=0,0,0,$(( 3000 + 5 ))
     73 
     74                  shape=sphere
     75                  boundary=Acrylic///$m1
     76                  parameters=0,0,0,$(( 3000 - 5 ))
     77 
     78                    )



viz
-------

::

    2016-10-31 20:46:50.716 INFO  [460591] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2016-10-31 20:46:50.716 INFO  [460591] [CTorchSource::GeneratePrimaryVertex@268] CTorchSource::GeneratePrimaryVertex typeName sphere modeString  position 0.0000,0.0000,0.0000 direction 0.0000,0.0000,1.0000 polarization 0.0000,0.0000,0.0000 radius 0 wavelength 430 time 0.1 polarization 0.0000,0.0000,0.0000 num 10000


* Polarization viz looks different in g4 and ok.
* Probably default G4 is random pol, and Opticks is some adhoc distrib... need to arrange these to match.

TODO: check polz distribs


truncation control
-------------------

::

    409    char bouncemax[128];
    410    snprintf(bouncemax,128,
    411 "Maximum number of boundary bounces, 0:prevents any propagation leaving generated photons"
    412 "Default %d ", m_bouncemax);
    413    m_desc.add_options()
    414        ("bouncemax,b",  boost::program_options::value<int>(&m_bouncemax), bouncemax );
    415 
    416 
    417    // keeping bouncemax one less than recordmax is advantageous 
    418    // as bookeeping is then consistent between the photons and the records 
    419    // as this avoiding truncation of the records
    420 
    421    char recordmax[128];
    422    snprintf(recordmax,128,
    423 "Maximum number of photon step records per photon, 1:to minimize without breaking machinery. Default %d ", m_recordmax);
    424    m_desc.add_options()
    425        ("recordmax,r",  boost::program_options::value<int>(&m_recordmax), recordmax );
    426 




Longstanding pflags issue
---------------------------

Rejoining in cfg4/CRecorder::RecordStepPoint was not scrubbing the AB in the mask on REjoining.

::

      .              pflags_ana  1:concentric   -1:concentric           c2           ab           ba 
                                    1000000      1000000     97516.50/48 = 2031.59 
       0                 1880        669935       670652             0.38        0.999 +- 0.001        1.001 +- 0.001  [3 ] TO|BT|SA
       1                 1008         83950        84177             0.31        0.997 +- 0.003        1.003 +- 0.003  [2 ] TO|AB
       2                 18a0         79964        80219             0.41        0.997 +- 0.004        1.003 +- 0.004  [4 ] TO|BT|SA|SC
       3                 1808         54175        54292             0.13        0.998 +- 0.004        1.002 +- 0.004  [3 ] TO|BT|AB
       4                 1890         38518            0         38518.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SA|RE
       5                 1898             0        37550         37550.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SA|RE|AB
       6                 1980         17805        17746             0.10        1.003 +- 0.008        0.997 +- 0.007  [4 ] TO|BT|DR|SA
       7                 1828          8738         8816             0.35        0.991 +- 0.011        1.009 +- 0.011  [4 ] TO|BT|SC|AB
       8                 1018          8204         7928             4.72        1.035 +- 0.011        0.966 +- 0.011  [3 ] TO|RE|AB
       9                 18b0          7928            0          7928.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SA|SC|RE
      10                 18b8             0         7780          7780.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|SA|SC|RE|AB
      11                 1818          6024         6081             0.27        0.991 +- 0.013        1.009 +- 0.013  [4 ] TO|BT|RE|AB
      12                 1908          5426         5491             0.39        0.988 +- 0.013        1.012 +- 0.014  [4 ] TO|BT|DR|AB
      13                 1028          5063         5064             0.00        1.000 +- 0.014        1.000 +- 0.014  [3 ] TO|SC|AB
      14                 19a0          4924         4960             0.13        0.993 +- 0.014        1.007 +- 0.014  [5 ] TO|BT|DR|SA|SC
      15                 1838          1525         1706            10.14        0.894 +- 0.023        1.119 +- 0.027  [5 ] TO|BT|SC|RE|AB
      16                 1990          1506            0          1506.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|DR|SA|RE
      17                 1998             0         1408          1408.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|DR|SA|RE|AB
      18                 1928          1062         1092             0.42        0.973 +- 0.030        1.028 +- 0.031  [5 ] TO|BT|DR|SC|AB
      19                 1918           619         1057           114.47        0.586 +- 0.024        1.708 +- 0.053  [5 ] TO|BT|DR|RE|AB
                                    1000000      1000000     97516.50/48 = 2031.59 


After rejoin scrubbing AB fix, some issues remain::

      .              pflags_ana  1:concentric   -1:concentric           c2           ab           ba 
                                    1000000      1000000       244.01/43 =  5.67 
       0                 1880        669935       670652             0.38        0.999 +- 0.001        1.001 +- 0.001  [3 ] TO|BT|SA
       1                 1008         83950        84177             0.31        0.997 +- 0.003        1.003 +- 0.003  [2 ] TO|AB
       2                 18a0         79964        80219             0.41        0.997 +- 0.004        1.003 +- 0.004  [4 ] TO|BT|SA|SC
       3                 1808         54175        54292             0.13        0.998 +- 0.004        1.002 +- 0.004  [3 ] TO|BT|AB
       4                 1890         38518        37550            12.32        1.026 +- 0.005        0.975 +- 0.005  [4 ] TO|BT|SA|RE
       5                 1980         17805        17746             0.10        1.003 +- 0.008        0.997 +- 0.007  [4 ] TO|BT|DR|SA
       6                 1828          8738         8816             0.35        0.991 +- 0.011        1.009 +- 0.011  [4 ] TO|BT|SC|AB
       7                 1018          8204         7928             4.72        1.035 +- 0.011        0.966 +- 0.011  [3 ] TO|RE|AB
       8                 18b0          7928         7780             1.39        1.019 +- 0.011        0.981 +- 0.011  [5 ] TO|BT|SA|SC|RE
       9                 1818          6024         6059             0.10        0.994 +- 0.013        1.006 +- 0.013  [4 ] TO|BT|RE|AB
      10                 1908          5426         5491             0.39        0.988 +- 0.013        1.012 +- 0.014  [4 ] TO|BT|DR|AB
      11                 1028          5063         5064             0.00        1.000 +- 0.014        1.000 +- 0.014  [3 ] TO|SC|AB
      12                 19a0          4924         4960             0.13        0.993 +- 0.014        1.007 +- 0.014  [5 ] TO|BT|DR|SA|SC
      13                 1838          1525         1462             1.33        1.043 +- 0.027        0.959 +- 0.025  [5 ] TO|BT|SC|RE|AB
      14                 1990          1506         1408             3.30        1.070 +- 0.028        0.935 +- 0.025  [5 ] TO|BT|DR|SA|RE
      15                 1928          1062         1092             0.42        0.973 +- 0.030        1.028 +- 0.031  [5 ] TO|BT|DR|SC|AB
      16                 1038           786          779             0.03        1.009 +- 0.036        0.991 +- 0.036  [4 ] TO|SC|RE|AB
      17                 1920           775          759             0.17        1.021 +- 0.037        0.979 +- 0.036  [4 ] TO|BT|DR|SC
      18                 1918           619          638             0.29        0.970 +- 0.039        1.031 +- 0.041  [5 ] TO|BT|DR|RE|AB
      19                 1910           482          419             4.41        1.150 +- 0.052        0.869 +- 0.042  [4 ] TO|BT|DR|RE
      20                 1930           455          412             2.13        1.104 +- 0.052        0.905 +- 0.045  [5 ] TO|BT|DR|SC|RE
      21                 1830           365          245            23.61        1.490 +- 0.078        0.671 +- 0.043  [4 ] TO|BT|SC|RE
      22                 19b0           301          300             0.00        1.003 +- 0.058        0.997 +- 0.058  [6 ] TO|BT|DR|SA|SC|RE
      23                 1ca0           213          263             5.25        0.810 +- 0.055        1.235 +- 0.076  [5 ] TO|BT|BR|SA|SC
      24                 1d80           204          166             3.90        1.229 +- 0.086        0.814 +- 0.063  [5 ] TO|BT|BR|DR|SA
      25                 1900           192          183             0.22        1.049 +- 0.076        0.953 +- 0.070  [3 ] TO|BT|DR
      26                 1820           170          136             3.78        1.250 +- 0.096        0.800 +- 0.069  [3 ] TO|BT|SC
      27                 1938           131          148             1.04        0.885 +- 0.077        1.130 +- 0.093  [6 ] TO|BT|DR|SC|RE|AB
      28                 1c20            95          119             2.69        0.798 +- 0.082        1.253 +- 0.115  [4 ] TO|BT|BR|SC
      29                 1c28            53          101            14.96        0.525 +- 0.072        1.906 +- 0.190  [5 ] TO|BT|BR|SC|AB

      30  ###            1888             0          100           100.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SA|AB
          ### BOTH SA and AB in same photon mask is impossible, as SA and AB both terminate .. 
          ### some bug here

      31                 1c90            66           82             1.73        0.805 +- 0.099        1.242 +- 0.137  [5 ] TO|BT|BR|SA|RE
      32                 1cb0            48           55             0.48        0.873 +- 0.126        1.146 +- 0.155  [6 ] TO|BT|BR|SA|SC|RE
      33                 1c10            39           52             1.86        0.750 +- 0.120        1.333 +- 0.185  [4 ] TO|BT|BR|RE
      34  ###            1c80             0           48            48.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|BR|SA
      35                 1da0            42           46             0.18        0.913 +- 0.141        1.095 +- 0.161  [6 ] TO|BT|BR|DR|SA|SC
      36                 1c18            35           31             0.24        1.129 +- 0.191        0.886 +- 0.159  [5 ] TO|BT|BR|RE|AB


Selecting just the seq that correspond to the funny mask, find no corresponding seq.  Bug in mask ? 

::

    simon:opticks blyth$ tconcentric.py --dbgmskhis 0x1888 --lmx 1000
    /Users/blyth/opticks/ana/tconcentric.py --dbgmskhis 0x1888 --lmx 1000
    [2016-11-02 17:26:48,469] p69790 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    [2016-11-02 17:26:48,469] p69790 {/Users/blyth/opticks/ana/evt.py:87} INFO -  dbgseqhis 0 dbgmskhis 1888 dbgseqmat 0 dbgmskmat 0 
    [2016-11-02 17:26:51,110] p69790 {/Users/blyth/opticks/ana/evt.py:87} INFO -  dbgseqhis 0 dbgmskhis 1888 dbgseqmat 0 dbgmskmat 0 
    CF a concentric/torch/  1 :  20161102-1517 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161102-1517 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    [2016-11-02 17:26:53,710] p69790 {/Users/blyth/opticks/ana/seq.py:361} INFO - compare dbgseq 0 dbgmsk 1888 
    .                seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000       706.24/348 =  2.03 
    .                               1000000      1000000       706.24/348 =  2.03 
    [2016-11-02 17:26:53,787] p69790 {/Users/blyth/opticks/ana/seq.py:361} INFO - compare dbgseq 1888 dbgmsk 0 
    .                pflags_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000       244.01/43 =  5.67 
      30                 1888             0          100           100.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SA|AB
      50                 18a8             0            5             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SA|SC|AB
      52                 1c98             0            3             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|BR|SA|RE|AB
      53                 1988             0            2             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|DR|SA|AB
      55                 19a8             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|DR|SA|SC|AB
      56                 1c88             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|BR|SA|AB
    .                               1000000      1000000       244.01/43 =  5.67 





looking for more zero counts
-------------------------------

Opticks is not doing "TO BT BT BT BR .."  

* no "internal" reflection in the acrylic just prior to MO in Opticks ?

::

      Gd/Ac/LS/Ac/MO


Dump only lines with zero counts, a rich source of bugs::

    simon:optickscore blyth$ tconcentric.py  --lmx 1000 --dbgzero
    /Users/blyth/opticks/ana/tconcentric.py --lmx 1000 --dbgzero
    [2016-11-02 13:41:46,821] p68095 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    CF a concentric/torch/  1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
                     seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                                    1000000      1000000       706.24/348 =  2.03 
     200        8cccccccbcccd             0           44            44.00        0.000 +- 0.000        0.000 +- 0.000  [13] TO BT BT BT BR BT BT BT BT BT BT BT SA
     466            4cccbcccd             0           11             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BT BR BT BT BT AB
     607      8cccc5555cc5ccd             0            7             0.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT RE BT BT RE RE RE RE BT BT BT BT SA
     642           4cccc5556d             7            0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC RE RE RE BT BT BT BT AB
     660     cccc5cc6cc9ccccd             0            6             0.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT SC BT BT RE BT BT BT BT
     689       8cccc9cccc655d             6            0             0.00        0.000 +- 0.000        0.000 +- 0.000  [14] TO RE RE SC BT BT BT BT DR BT BT BT BT SA
     697         8cc6cc55555d             0            6             0.00        0.000 +- 0.000        0.000 +- 0.000  [12] TO RE RE RE RE RE BT BT SC BT BT SA
     734     cc6cccccc96ccccd             5            0             0.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT SC DR BT BT BT BT BT BT SC BT BT
     757         4cccccc65ccd             0            5             0.00        0.000 +- 0.000        0.000 +- 0.000  [12] TO BT BT RE SC BT BT BT BT BT BT AB
     796       8cccc6cccbcccd             0            5             0.00        0.000 +- 0.000        0.000 +- 0.000  [14] TO BT BT BT BR BT BT BT SC BT BT BT BT SA
     800             4c555ccd             0            5             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT RE RE RE BT AB


Dump only lines starting "TO BT BT BT BR"::

    simon:optickscore blyth$ tconcentric.py --dbgseqhis bcccd
    /Users/blyth/opticks/ana/tconcentric.py --dbgseqhis bcccd
    [2016-11-02 13:47:51,381] p68110 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    CF a concentric/torch/  1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
                     seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                                    1000000      1000000       706.24/348 =  2.03 
     200        8cccccccbcccd             0           44            44.00        0.000 +- 0.000        0.000 +- 0.000  [13] TO BT BT BT BR BT BT BT BT BT BT BT SA
     466            4cccbcccd             0           11             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BT BR BT BT BT AB
     796       8cccc6cccbcccd             0            5             0.00        0.000 +- 0.000        0.000 +- 0.000  [14] TO BT BT BT BR BT BT BT SC BT BT BT BT SA
    1052           45cccbcccd             0            3             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BT RE AB
    1176       8cccc5cccbcccd             0            3             0.00        0.000 +- 0.000        0.000 +- 0.000  [14] TO BT BT BT BR BT BT BT RE BT BT BT BT SA
    2198       89cccccccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [14] TO BT BT BT BR BT BT BT BT BT BT BT DR SA
    2225         4cc6cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [12] TO BT BT BT BR BT BT BT SC BT BT AB
    2474     ccc55cc5cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BR BT BT BT RE BT BT RE RE BT BT BT
    2521          466cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [11] TO BT BT BT BR BT BT BT SC SC AB
    2812           46cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BT SC AB
    2961      86cccc5cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BR BT BT BT RE BT BT BT BT SC SA
    2995        4cccccccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [13] TO BT BT BT BR BT BT BT BT BT BT BT AB
    3244      8cccc55cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BR BT BT BT RE RE BT BT BT BT SA
    3330     cccc5555cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BR BT BT BT RE RE RE RE BT BT BT BT
    3798     89cccccc55cbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BR BT RE RE BT BT BT BT BT BT DR SA
    3954       8cccccc5cbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [14] TO BT BT BT BR BT RE BT BT BT BT BT BT SA
    4056      8cc56cccccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BR BT BT BT BT BT SC RE BT BT SA
    4165           8cc5cbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT RE BT BT SA
    4178     8cccc555cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BR BT BT BT RE RE RE BT BT BT BT SA
    4526          456cccbcccd             0            1             0.00        0.000 +- 0.000        0.000 +- 0.000  [11] TO BT BT BT BR BT BT BT SC RE AB
                                    1000000      1000000       706.24/348 =  2.03 


remaining discreps by cmx selection
--------------------------------------

::

    23: truncation diff
    200: opticks zero 

    simon:optickscore blyth$ tconcentric.py  --lmx 500 --cmx 5
    /Users/blyth/opticks/ana/tconcentric.py --lmx 500 --cmx 5
    [2016-11-02 13:57:06,819] p68134 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    CF a concentric/torch/  1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
                     seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                                    1000000      1000000       706.24/348 =  2.03 
       5              8cccc5d         20239        19674             8.00        1.029 +- 0.007        0.972 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              8cc6ccd         10397        10919            12.78        0.952 +- 0.009        1.050 +- 0.010  [7 ] TO BT BT SC BT BT SA
       7              86ccccd         10160        10825            21.07        0.939 +- 0.009        1.065 +- 0.010  [7 ] TO BT BT BT BT SC SA
      15          8cccccc6ccd          3190         2785            27.45        1.145 +- 0.020        0.873 +- 0.017  [11] TO BT BT SC BT BT BT BT BT BT SA
      20          8cccc6ccccd          1580         1805            14.96        0.875 +- 0.022        1.142 +- 0.027  [11] TO BT BT BT BT SC BT BT BT BT SA
      23      8cccccccc6ccccd          1577          998           130.19        1.580 +- 0.040        0.633 +- 0.020  [15] TO BT BT BT BT SC BT BT BT BT BT BT BT BT SA
      36              46ccccd           788          977            20.24        0.807 +- 0.029        1.240 +- 0.040  [7 ] TO BT BT BT BT SC AB
      51          4cccc6ccccd           384          308             8.35        1.247 +- 0.064        0.802 +- 0.046  [11] TO BT BT BT BT SC BT BT BT BT AB
      72             89cccc5d           234          183             6.24        1.279 +- 0.084        0.782 +- 0.058  [8 ] TO RE BT BT BT BT DR SA
      88     8cccc6cccc6ccccd           143           98             8.40        1.459 +- 0.122        0.685 +- 0.069  [16] TO BT BT BT BT SC BT BT BT BT SC BT BT BT BT SA
      90            8cc55cc5d           136          100             5.49        1.360 +- 0.117        0.735 +- 0.074  [9 ] TO RE BT BT RE RE BT BT SA
      98         8cccc5cc6ccd           122           83             7.42        1.470 +- 0.133        0.680 +- 0.075  [12] TO BT BT SC BT BT RE BT BT BT BT SA
     102            8cccc565d            84          119             6.03        0.706 +- 0.077        1.417 +- 0.130  [9 ] TO RE SC RE BT BT BT BT SA
     105     8cccccccc6cccc6d           116           67            13.12        1.731 +- 0.161        0.578 +- 0.071  [16] TO SC BT BT BT BT SC BT BT BT BT BT BT BT BT SA
     123              8c6cccd            89           60             5.64        1.483 +- 0.157        0.674 +- 0.087  [7 ] TO BT BT BT SC BT SA
     142        4cccccc6ccccd            75           46             6.95        1.630 +- 0.188        0.613 +- 0.090  [13] TO BT BT BT BT SC BT BT BT BT BT BT AB
     157     8cccc5cccc6ccccd            67           36             9.33        1.861 +- 0.227        0.537 +- 0.090  [16] TO BT BT BT BT SC BT BT BT BT RE BT BT BT BT SA
     159         8cccc66ccccd            35           66             9.51        0.530 +- 0.090        1.886 +- 0.232  [12] TO BT BT BT BT SC SC BT BT BT BT SA
     200        8cccccccbcccd             0           44            44.00        0.000 +- 0.000        0.000 +- 0.000  [13] TO BT BT BT BR BT BT BT BT BT BT BT SA
     225           8cccbc5ccd            17           36             6.81        0.472 +- 0.115        2.118 +- 0.353  [10] TO BT BT RE BT BR BT BT BT SA
     236            86cccc56d            17           33             5.12        0.515 +- 0.125        1.941 +- 0.338  [9 ] TO SC RE BT BT BT BT SC SA
     275           4cc6cc6ccd            27           12             5.77        2.250 +- 0.433        0.444 +- 0.128  [10] TO BT BT SC BT BT SC BT BT AB
                                    1000000      1000000       706.24/348 =  2.03 



truncation avoidance trick
----------------------------

Arranging maxbounce to be one less than maxrec avoids much of the truncation discrepancy

::

    simon:optickscore blyth$ tconcentric.py  --lmx 500 
    /Users/blyth/opticks/ana/tconcentric.py --lmx 500
    [2016-11-02 13:37:40,224] p68080 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    CF a concentric/torch/  1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161102-1256 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
                     seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                                    1000000      1000000       706.24/348 =  2.03 
       0               8ccccd        669935       670752             0.50        0.999 +- 0.001        1.001 +- 0.001  [6 ] TO BT BT BT BT SA
       1                   4d         83950        84177             0.31        0.997 +- 0.003        1.003 +- 0.003  [2 ] TO AB
       2              8cccc6d         45599        45475             0.17        1.003 +- 0.005        0.997 +- 0.005  [7 ] TO SC BT BT BT BT SA
       3               4ccccd         28958        28871             0.13        1.003 +- 0.006        0.997 +- 0.006  [6 ] TO BT BT BT BT AB
       4                 4ccd         23187        23447             1.45        0.989 +- 0.006        1.011 +- 0.007  [4 ] TO BT BT AB
       5              8cccc5d         20239        19674             8.00        1.029 +- 0.007        0.972 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              8cc6ccd         10397        10919            12.78        0.952 +- 0.009        1.050 +- 0.010  [7 ] TO BT BT SC BT BT SA
       7              86ccccd         10160        10825            21.07        0.939 +- 0.009        1.065 +- 0.010  [7 ] TO BT BT BT BT SC SA
       8              89ccccd          7605         7685             0.42        0.990 +- 0.011        1.011 +- 0.012  [7 ] TO BT BT BT BT DR SA
       9             8cccc55d          5970         5911             0.29        1.010 +- 0.013        0.990 +- 0.013  [8 ] TO RE RE BT BT BT BT SA
      10                  45d          5780         5627             2.05        1.027 +- 0.014        0.974 +- 0.013  [3 ] TO RE AB
      11      8cccccccc9ccccd          5350         5289             0.35        1.012 +- 0.014        0.989 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      12              8cc5ccd          5113         4948             2.71        1.033 +- 0.014        0.968 +- 0.014  [7 ] TO BT BT RE BT BT SA
      13                  46d          4783         4808             0.07        0.995 +- 0.014        1.005 +- 0.014  [3 ] TO SC AB
      14          8cccc9ccccd          4525         4452             0.59        1.016 +- 0.015        0.984 +- 0.015  [11] TO BT BT BT BT DR BT BT BT BT SA
      15          8cccccc6ccd          3190         2785            27.45        1.145 +- 0.020        0.873 +- 0.017  [11] TO BT BT SC BT BT BT BT BT BT SA
      16             8cccc66d          2600         2642             0.34        0.984 +- 0.019        1.016 +- 0.020  [8 ] TO SC SC BT BT BT BT SA
      17              49ccccd          2313         2452             4.05        0.943 +- 0.020        1.060 +- 0.021  [7 ] TO BT BT BT BT DR AB
      18              4cccc6d          2027         2040             0.04        0.994 +- 0.022        1.006 +- 0.022  [7 ] TO SC BT BT BT BT AB
      19            8cccc555d          1819         1696             4.30        1.073 +- 0.025        0.932 +- 0.023  [9 ] TO RE RE RE BT BT BT BT SA
      20          8cccc6ccccd          1580         1805            14.96        0.875 +- 0.022        1.142 +- 0.027  [11] TO BT BT BT BT SC BT BT BT BT SA
      21                4cc6d          1733         1792             0.99        0.967 +- 0.023        1.034 +- 0.024  [5 ] TO SC BT BT AB
      22                 455d          1695         1619             1.74        1.047 +- 0.025        0.955 +- 0.024  [4 ] TO RE RE AB
      23      8cccccccc6ccccd          1577          998           130.19        1.580 +- 0.040        0.633 +- 0.020  [15] TO BT BT BT BT SC BT BT BT BT BT BT BT BT SA
      24          4cccc9ccccd          1310         1257             1.09        1.042 +- 0.029        0.960 +- 0.027  [11] TO BT BT BT BT DR BT BT BT BT AB
      25             8cc55ccd          1268         1262             0.01        1.005 +- 0.028        0.995 +- 0.028  [8 ] TO BT BT RE RE BT BT SA
      26             8cccc56d          1170         1104             1.92        1.060 +- 0.031        0.944 +- 0.028  [8 ] TO SC RE BT BT BT BT SA
      27                45ccd          1168         1090             2.69        1.072 +- 0.031        0.933 +- 0.028  [5 ] TO BT BT RE AB
      28          8cccccc5ccd          1104         1157             1.24        0.954 +- 0.029        1.048 +- 0.031  [11] TO BT BT RE BT BT BT BT BT BT SA
      29              4cc6ccd          1148         1045             4.84        1.099 +- 0.032        0.910 +- 0.028  [7 ] TO BT BT SC BT BT AB
      30             8cccc65d          1133         1066             2.04        1.063 +- 0.032        0.941 +- 0.029  [8 ] TO RE SC BT BT BT BT SA
      31                  4cd          1035         1056             0.21        0.980 +- 0.030        1.020 +- 0.031  [3 ] TO BT AB
      32            4cc9ccccd          1048         1048             0.00        1.000 +- 0.031        1.000 +- 0.031  [9 ] TO BT BT BT BT DR BT BT AB
      33                4cc5d          1036         1018             0.16        1.018 +- 0.032        0.983 +- 0.031  [5 ] TO RE BT BT AB
      34              4cccc5d           965         1023             1.69        0.943 +- 0.030        1.060 +- 0.033  [7 ] TO RE BT BT BT BT AB
      35                4cccd           995          918             3.10        1.084 +- 0.034        0.923 +- 0.030  [5 ] TO BT BT BT AB
      36              46ccccd           788          977            20.24        0.807 +- 0.029        1.240 +- 0.040  [7 ] TO BT BT BT BT SC AB
      37             869ccccd           915          893             0.27        1.025 +- 0.034        0.976 +- 0.033  [8 ] TO BT BT BT BT DR SC SA
      38             8cc6cc6d           803          809             0.02        0.993 +- 0.035        1.007 +- 0.035  [8 ] TO SC BT BT SC BT BT SA
      39             86cccc6d           725          764             1.02        0.949 +- 0.035        1.054 +- 0.038  [8 ] TO SC BT BT BT BT SC SA
      40                46ccd           635          689             2.20        0.922 +- 0.037        1.085 +- 0.041  [5 ] TO BT BT SC AB
      41              4cc5ccd           611          678             3.48        0.901 +- 0.036        1.110 +- 0.043  [7 ] TO BT BT RE BT BT AB
      42     8cccc6cccc9ccccd           571          575             0.01        0.993 +- 0.042        1.007 +- 0.042  [16] TO BT BT BT BT DR BT BT BT BT SC BT BT BT BT SA





zeros in long tail issue, from truncation ?
---------------------------------------------

::

    simon:ana blyth$ t tconcentric-tt
    tconcentric-tt () 
    { 
        tconcentric-t --bouncemax 15 --recordmax 15 $*
    }



::

    simon:ana blyth$ tconcentric.py --lmx 300 --dbgzero
    /Users/blyth/opticks/ana/tconcentric.py --lmx 300 --dbgzero
    [2016-11-02 12:21:32,078] p63764 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    [2016-11-02 12:21:37,011] p63764 {/Users/blyth/opticks/ana/cf.py:39} INFO - CF a concentric/torch/  1 :  20161101-2009 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-02 12:21:37,011] p63764 {/Users/blyth/opticks/ana/cf.py:40} INFO - CF b concentric/torch/ -1 :  20161101-2009 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
      42      8ccc6cccc9ccccd             0          575           575.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT SC BT BT BT SA
      43     8cccc6cccc9ccccd           571            0           571.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT SC BT BT BT BT SA
      50      8ccccccc9cccc6d             0          398           398.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO SC BT BT BT BT DR BT BT BT BT BT BT BT SA
      53     8cccccccc9cccc6d           369            0           369.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO SC BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      73     8cccc5cccc9ccccd           236            0           236.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT RE BT BT BT BT SA
      76      8ccc5cccc9ccccd             0          204           204.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT RE BT BT BT SA
      80      ccccccccc9ccccd             0          172           172.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT BT
      90      8ccccccc9cccc5d             0          144           144.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO RE BT BT BT BT DR BT BT BT BT BT BT BT SA
      91     c9cccccccc9ccccd           144            0           144.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT BT BT BT BT DR BT
      92     8cccc6cccc6ccccd           143            0           143.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT SC BT BT BT BT SC BT BT BT BT SA
      96     8cccccccc9cccc5d           135            0           135.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO RE BT BT BT BT DR BT BT BT BT BT BT BT BT SA
     105      8c6cccccc9ccccd             0          120           120.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT BT BT SC BT SA
     110     8cccccccc6cccc6d           116            0           116.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO SC BT BT BT BT SC BT BT BT BT BT BT BT BT SA
     112      8ccccc6cc9ccccd             0          114           114.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT SC BT BT BT BT BT SA
     116     8cccccc6cc9ccccd           107            0           107.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT SC BT BT BT BT BT BT SA
     120     8cc6cccccc9ccccd           103            0           103.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT BT BT SC BT BT SA
     121     86cccccccc9ccccd           102            0           102.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SC SA
     124      8ccc6cccc6ccccd             0           98            98.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT SC BT BT BT BT SC BT BT BT SA
     125      8ccccccc69ccccd             0           98            98.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR SC BT BT BT BT BT BT BT SA
     133      ccc55cccc9ccccd             0           88            88.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT RE RE BT BT BT
     136      8ccccccc96ccccd             0           85            85.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT SC DR BT BT BT BT BT BT BT SA
     137      8ccccccc9cc6ccd             0           84            84.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT SC BT BT DR BT BT BT BT BT BT BT SA
     139     8cccccccc69ccccd            83            0            83.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR SC BT BT BT BT BT BT BT BT SA
     144     cccc55cccc9ccccd            80            0            80.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT RE RE BT BT BT BT
     146     8cccccccc9cc6ccd            80            0            80.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT SC BT BT DR BT BT BT BT BT BT BT BT SA
     167      8c5cccccc9ccccd             0           68            68.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT BT BT RE BT SA
     168     8cccc5cccc6ccccd            67            0            67.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT SC BT BT BT BT RE BT BT BT BT SA







::

    simon:ana blyth$ tconcentric.py --lmx 300
    /Users/blyth/opticks/ana/tconcentric.py --lmx 300
    [2016-11-02 11:45:54,982] p63687 {/Users/blyth/opticks/ana/tconcentric.py:24} INFO - tag 1 src torch det concentric c2max 2.0  
    [2016-11-02 11:45:59,898] p63687 {/Users/blyth/opticks/ana/cf.py:38} INFO - CF a concentric/torch/  1 :  20161101-2009 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-02 11:45:59,898] p63687 {/Users/blyth/opticks/ana/cf.py:39} INFO - CF b concentric/torch/ -1 :  20161101-2009 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
       0               8ccccd        669935       670752             0.50        0.999 +- 0.001        1.001 +- 0.001  [6 ] TO BT BT BT BT SA
       1                   4d         83950        84177             0.31        0.997 +- 0.003        1.003 +- 0.003  [2 ] TO AB
       2              8cccc6d         45599        45475             0.17        1.003 +- 0.005        0.997 +- 0.005  [7 ] TO SC BT BT BT BT SA
       3               4ccccd         28958        28871             0.13        1.003 +- 0.006        0.997 +- 0.006  [6 ] TO BT BT BT BT AB
       4                 4ccd         23187        23447             1.45        0.989 +- 0.006        1.011 +- 0.007  [4 ] TO BT BT AB
       5              8cccc5d         20239        19674             8.00        1.029 +- 0.007        0.972 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              8cc6ccd         10397        10919            12.78        0.952 +- 0.009        1.050 +- 0.010  [7 ] TO BT BT SC BT BT SA
       7              86ccccd         10160        10825            21.07        0.939 +- 0.009        1.065 +- 0.010  [7 ] TO BT BT BT BT SC SA
       8              89ccccd          7605         7685             0.42        0.990 +- 0.011        1.011 +- 0.012  [7 ] TO BT BT BT BT DR SA

      ...  selecting the big c2 contributors, are being killed by zeros in tail ... possibly truncation related
     # Opticks reporting extra BT ??? Or CFG4 missing a BT (prior to SA, or after DR???)
       
      42      8ccc6cccc9ccccd             0          575           575.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT SC BT BT BT SA
      43     8cccc6cccc9ccccd           571            0           571.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT SC BT BT BT BT SA

      50      8ccccccc9cccc6d             0          398           398.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO SC BT BT BT BT DR BT BT BT BT BT BT BT SA
      53     8cccccccc9cccc6d           369            0           369.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO SC BT BT BT BT DR BT BT BT BT BT BT BT BT SA

      76      8ccc5cccc9ccccd             0          204           204.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT RE BT BT BT SA
      73     8cccc5cccc9ccccd           236            0           236.00        0.000 +- 0.000        0.000 +- 0.000  [16] TO BT BT BT BT DR BT BT BT BT RE BT BT BT BT SA


      80      ccccccccc9ccccd             0          172           172.00        0.000 +- 0.000        0.000 +- 0.000  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT BT
        # trunc without counterpart ??


                             1000000      1000000      6683.62/355 = 18.83 




with flags fixed
-----------------

Perhaps an SC discrep ? 

::


    [2016-11-01 21:09:04,278] p62207 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161101-2009 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-01 21:09:04,278] p62207 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161101-2009 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    [2016-11-01 21:09:04,288] p62207 {/Users/blyth/opticks/ana/seq.py:279} INFO - SeqTable.compare forming cf ad.code len(u) 5099 
    [2016-11-01 21:09:04,353] p62207 {/Users/blyth/opticks/ana/seq.py:284} INFO - SeqTable.compare forming cf af.code DONE 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd        669935       670752             0.50         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d         83950        84177             0.31         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO AB
                 8cccc6d         45599        45475             0.17         1.00 +- 0.00         1.00 +- 0.00  [7 ] TO SC BT BT BT BT SA
                  4ccccd         28958        28871             0.13         1.00 +- 0.01         1.00 +- 0.01  [6 ] TO BT BT BT BT AB
                    4ccd         23187        23447             1.45         0.99 +- 0.01         1.01 +- 0.01  [4 ] TO BT BT AB
                 8cccc5d         20239        19674             8.00         1.03 +- 0.01         0.97 +- 0.01  [7 ] TO RE BT BT BT BT SA
         ##      8cc6ccd         10397        10919            12.78         0.95 +- 0.01         1.05 +- 0.01  [7 ] TO BT BT SC BT BT SA
         ##      86ccccd         10160        10825            21.07         0.94 +- 0.01         1.07 +- 0.01  [7 ] TO BT BT BT BT SC SA
                 89ccccd          7605         7685             0.42         0.99 +- 0.01         1.01 +- 0.01  [7 ] TO BT BT BT BT DR SA
                8cccc55d          5970         5911             0.29         1.01 +- 0.01         0.99 +- 0.01  [8 ] TO RE RE BT BT BT BT SA
                     45d          5780         5627             2.05         1.03 +- 0.01         0.97 +- 0.01  [3 ] TO RE AB
         8cccccccc9ccccd          5350         5451             0.94         0.98 +- 0.01         1.02 +- 0.01  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
                 8cc5ccd          5113         4948             2.71         1.03 +- 0.01         0.97 +- 0.01  [7 ] TO BT BT RE BT BT SA
                     46d          4783         4808             0.07         0.99 +- 0.01         1.01 +- 0.01  [3 ] TO SC AB
             8cccc9ccccd          4525         4452             0.59         1.02 +- 0.02         0.98 +- 0.01  [11] TO BT BT BT BT DR BT BT BT BT SA
         ##  8cccccc6ccd          3190         2785            27.45         1.15 +- 0.02         0.87 +- 0.02  [11] TO BT BT SC BT BT BT BT BT BT SA
                8cccc66d          2600         2642             0.34         0.98 +- 0.02         1.02 +- 0.02  [8 ] TO SC SC BT BT BT BT SA
                 49ccccd          2313         2452             4.05         0.94 +- 0.02         1.06 +- 0.02  [7 ] TO BT BT BT BT DR AB
                 4cccc6d          2027         2040             0.04         0.99 +- 0.02         1.01 +- 0.02  [7 ] TO SC BT BT BT BT AB
               8cccc555d          1819         1696             4.30         1.07 +- 0.03         0.93 +- 0.02  [9 ] TO RE RE RE BT BT BT BT SA
                             1000000      1000000        18.83 




FIXED : funny badflag 00
----------------------------

Using "--bouncemax 15 --recordmax 16" is causing some badflag zeros at python level only

* could be bumping into signbit somewhere
* nope count_unique was going via float and loosing precision with large uint64 see ana/nbase.py:test_count_unique



avoiding truncation by pushing bouncemax and recordmax
--------------------------------------------------------

::

    tconcentric-tt () 
    { 
        tconcentric-t --bouncemax 15 --recordmax 16 $*
    }


::

        .    seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd        669935       670752             0.50         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d         83950        84177             0.31         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO AB
                 8cccc6d         45599        45475             0.17         1.00 +- 0.00         1.00 +- 0.00  [7 ] TO SC BT BT BT BT SA
                  4ccccd         28958        28871             0.13         1.00 +- 0.01         1.00 +- 0.01  [6 ] TO BT BT BT BT AB
                    4ccd         23187        23447             1.45         0.99 +- 0.01         1.01 +- 0.01  [4 ] TO BT BT AB
                 8cccc5d         20239        19674             8.00         1.03 +- 0.01         0.97 +- 0.01  [7 ] TO RE BT BT BT BT SA
                 8cc6ccd         10397        10919            12.78         0.95 +- 0.01         1.05 +- 0.01  [7 ] TO BT BT SC BT BT SA   ##
                 86ccccd         10160        10825            21.07         0.94 +- 0.01         1.07 +- 0.01  [7 ] TO BT BT BT BT SC SA   ##
                 89ccccd          7605         7685             0.42         0.99 +- 0.01         1.01 +- 0.01  [7 ] TO BT BT BT BT DR SA
                8cccc55d          5970         5911             0.29         1.01 +- 0.01         0.99 +- 0.01  [8 ] TO RE RE BT BT BT BT SA
                     45d          5780         5627             2.05         1.03 +- 0.01         0.97 +- 0.01  [3 ] TO RE AB
         8cccccccc9ccd00          5350         5289             0.35         1.01 +- 0.01         0.99 +- 0.01  [15] ?0? ?0? TO BT BT DR BT BT BT BT BT BT BT BT SA
                 8cc5ccd          5113         4948             2.71         1.03 +- 0.01         0.97 +- 0.01  [7 ] TO BT BT RE BT BT SA
                     46d          4783         4808             0.07         0.99 +- 0.01         1.01 +- 0.01  [3 ] TO SC AB
             8cccc9ccccd          4525         4452             0.59         1.02 +- 0.02         0.98 +- 0.01  [11] TO BT BT BT BT DR BT BT BT BT SA
             8cccccc6ccd          3190         2785            27.45         1.15 +- 0.02         0.87 +- 0.02  [11] TO BT BT SC BT BT BT BT BT BT SA   ##
                8cccc66d          2600         2642             0.34         0.98 +- 0.02         1.02 +- 0.02  [8 ] TO SC SC BT BT BT BT SA
                 49ccccd          2313         2452             4.05         0.94 +- 0.02         1.06 +- 0.02  [7 ] TO BT BT BT BT DR AB
                 4cccc6d          2027         2040             0.04         0.99 +- 0.02         1.01 +- 0.02  [7 ] TO SC BT BT BT BT AB
               8cccc555d          1819         1696             4.30         1.07 +- 0.03         0.93 +- 0.02  [9 ] TO RE RE RE BT BT BT BT SA
                              999037       999146         2.02 




switch to unified model fixes incorrectly specular DR
--------------------------------------------------------

* push stats to 1M,  biggest discreps from trunc, possible from SC also  

::

    [2016-11-01 19:01:44,889] p60660 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161101-1901 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-01 19:01:44,889] p60660 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161101-1901 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd        669935       671359             1.51         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d         83950        84265             0.59         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO AB
                 8cccc6d         45599        45244             1.39         1.01 +- 0.00         0.99 +- 0.00  [7 ] TO SC BT BT BT BT SA
                  4ccccd         28958        28822             0.32         1.00 +- 0.01         1.00 +- 0.01  [6 ] TO BT BT BT BT AB
                    4ccd         23187        23366             0.69         0.99 +- 0.01         1.01 +- 0.01  [4 ] TO BT BT AB
                 8cccc5d         20239        19818             4.42         1.02 +- 0.01         0.98 +- 0.01  [7 ] TO RE BT BT BT BT SA
              cccc9ccccd         14235        13343            28.85         1.07 +- 0.01         0.94 +- 0.01  [10] TO BT BT BT BT DR BT BT BT BT   ###  trunc
                 8cc6ccd         10397        10930            13.32         0.95 +- 0.01         1.05 +- 0.01  [7 ] TO BT BT SC BT BT SA            ##  SC 
                 86ccccd         10160        10725            15.28         0.95 +- 0.01         1.06 +- 0.01  [7 ] TO BT BT BT BT SC SA            ##  SC
                 89ccccd          7605         7600             0.00         1.00 +- 0.01         1.00 +- 0.01  [7 ] TO BT BT BT BT DR SA
                8cccc55d          5970         5847             1.28         1.02 +- 0.01         0.98 +- 0.01  [8 ] TO RE RE BT BT BT BT SA
                     45d          5780         5616             2.36         1.03 +- 0.01         0.97 +- 0.01  [3 ] TO RE AB
                 8cc5ccd          5113         4972             1.97         1.03 +- 0.01         0.97 +- 0.01  [7 ] TO BT BT RE BT BT SA
                     46d          4783         4847             0.43         0.99 +- 0.01         1.01 +- 0.01  [3 ] TO SC AB
              cccc6ccccd          4404         3549            91.92         1.24 +- 0.02         0.81 +- 0.01  [10] TO BT BT BT BT SC BT BT BT BT   ### trunc
              cccccc6ccd          3588         3158            27.41         1.14 +- 0.02         0.88 +- 0.02  [10] TO BT BT SC BT BT BT BT BT BT   ### trunc
                8cccc66d          2600         2641             0.32         0.98 +- 0.02         1.02 +- 0.02  [8 ] TO SC SC BT BT BT BT SA
                 49ccccd          2313         2425             2.65         0.95 +- 0.02         1.05 +- 0.02  [7 ] TO BT BT BT BT DR AB
                 4cccc6d          2027         2054             0.18         0.99 +- 0.02         1.01 +- 0.02  [7 ] TO SC BT BT BT BT AB
               8cccc555d          1819         1684             5.20         1.08 +- 0.03         0.93 +- 0.02  [9 ] TO RE RE RE BT BT BT BT SA
                             1000000      1000000         4.22 



* remaining low level discrep from scattering in MO ?

::

    [2016-11-01 18:54:04,720] p60414 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161101-1853 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-01 18:54:04,720] p60414 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161101-1853 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd         67144        67082             0.03         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d          8398         8355             0.11         1.01 +- 0.01         0.99 +- 0.01  [2 ] TO AB
                 8cccc6d          4567         4564             0.00         1.00 +- 0.01         1.00 +- 0.01  [7 ] TO SC BT BT BT BT SA
                  4ccccd          2912         2935             0.09         0.99 +- 0.02         1.01 +- 0.02  [6 ] TO BT BT BT BT AB
                    4ccd          2264         2369             2.38         0.96 +- 0.02         1.05 +- 0.02  [4 ] TO BT BT AB
                 8cccc5d          2056         1994             0.95         1.03 +- 0.02         0.97 +- 0.02  [7 ] TO RE BT BT BT BT SA
              cccc9ccccd          1402         1311             3.05         1.07 +- 0.03         0.94 +- 0.03  [10] TO BT BT BT BT DR BT BT BT BT
                 86ccccd           961         1109            10.58         0.87 +- 0.03         1.15 +- 0.03  [7 ] TO BT BT BT BT SC SA             ## SC in MO ??? more in cfg4
                 8cc6ccd          1035         1091             1.48         0.95 +- 0.03         1.05 +- 0.03  [7 ] TO BT BT SC BT BT SA
                 89ccccd           711          783             3.47         0.91 +- 0.03         1.10 +- 0.04  [7 ] TO BT BT BT BT DR SA
                8cccc55d           601          616             0.18         0.98 +- 0.04         1.02 +- 0.04  [8 ] TO RE RE BT BT BT BT SA
                     45d           554          537             0.26         1.03 +- 0.04         0.97 +- 0.04  [3 ] TO RE AB
                     46d           539          484             2.96         1.11 +- 0.05         0.90 +- 0.04  [3 ] TO SC AB
                 8cc5ccd           485          478             0.05         1.01 +- 0.05         0.99 +- 0.05  [7 ] TO BT BT RE BT BT SA
              cccc6ccccd           461          357            13.22         1.29 +- 0.06         0.77 +- 0.04  [10] TO BT BT BT BT SC BT BT BT BT    ##  SC in MO ??? less in cfg4
              cccccc6ccd           346          350             0.02         0.99 +- 0.05         1.01 +- 0.05  [10] TO BT BT SC BT BT BT BT BT BT
                8cccc66d           280          254             1.27         1.10 +- 0.07         0.91 +- 0.06  [8 ] TO SC SC BT BT BT BT SA
                 49ccccd           207          234             1.65         0.88 +- 0.06         1.13 +- 0.07  [7 ] TO BT BT BT BT DR AB
                 4cccc6d           218          200             0.78         1.09 +- 0.07         0.92 +- 0.06  [7 ] TO SC BT BT BT BT AB
                   4cc6d           194          181             0.45         1.07 +- 0.08         0.93 +- 0.07  [5 ] TO SC BT BT AB
                              100000       100000         1.98 







post DR discrep
----------------

tconcentric-v and selecting 89ccccd  "TO BT BT BT BT DR SA" shows some extreme DR kinks for Opticks.

Checking in tconcentric_distrib.py for related category 49ccccd "TO BT BT BT BT DR AB"
shows that cfg4 DR is happening specularly, with all the AB being along the incoming line. 
So for cfg4 DR behaves like BR.


::

    tconcentric.py --dbgseqhis 9ccccd

          seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
          cccc9ccccd          1396         2267           207.11         0.62 +- 0.02         1.62 +- 0.03  [10] TO BT BT BT BT DR BT BT BT BT
             89ccccd           725            0           725.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR SA
             49ccccd           221          115            33.44         1.92 +- 0.13         0.52 +- 0.05  [7 ] TO BT BT BT BT DR AB
          5ccc9ccccd             0          201           201.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT RE
           4cc9ccccd            83           79             0.10         1.05 +- 0.12         0.95 +- 0.11  [9 ] TO BT BT BT BT DR BT BT AB
            869ccccd            77           36            14.88         2.14 +- 0.24         0.47 +- 0.08  [8 ] TO BT BT BT BT DR SC SA
          c6cc9ccccd            75           45             7.50         1.67 +- 0.19         0.60 +- 0.09  [10] TO BT BT BT BT DR BT BT SC BT
          c5cc9ccccd            34           21             3.07         1.62 +- 0.28         0.62 +- 0.13  [10] TO BT BT BT BT DR BT BT RE BT
          ccc69ccccd            32           19             3.31         1.68 +- 0.30         0.59 +- 0.14  [10] TO BT BT BT BT DR SC BT BT BT
          ccc99ccccd            19            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR DR BT BT BT
          55cc9ccccd            12            8             0.00         1.50 +- 0.43         0.67 +- 0.24  [10] TO BT BT BT BT DR BT BT RE RE
            899ccccd            10            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR DR SA
            469ccccd             5           10             0.00         0.50 +- 0.22         2.00 +- 0.63  [8 ] TO BT BT BT BT DR SC AB
          bccc9ccccd             9            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT BR
          45cc9ccccd             8            2             0.00         4.00 +- 1.41         0.25 +- 0.18  [10] TO BT BT BT BT DR BT BT RE AB
            4c9ccccd             8            4             0.00         2.00 +- 0.71         0.50 +- 0.25  [8 ] TO BT BT BT BT DR BT AB
            8b9ccccd             7            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR BR SA
          46cc9ccccd             6            3             0.00         2.00 +- 0.82         0.50 +- 0.29  [10] TO BT BT BT BT DR BT BT SC AB
          4ccc9ccccd             6            2             0.00         3.00 +- 1.22         0.33 +- 0.24  [10] TO BT BT BT BT DR BT BT BT AB
            499ccccd             5            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR DR AB
                          100000       100000        17.81 


seqhis after optical surfaces hookup with test geometry
----------------------------------------------------------

With single direction source (like tlaser) behavior very similar, as geometry is symmetrical. But this is easier for 
some distrib angle comparisons

::

    [2016-11-01 17:54:40,804] p58713 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161101-1754 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-01 17:54:40,804] p58713 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161101-1754 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd         67144        67106             0.01         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d          8398         8316             0.40         1.01 +- 0.01         0.99 +- 0.01  [2 ] TO AB
                 8cccc6d          4567         4612             0.22         0.99 +- 0.01         1.01 +- 0.01  [7 ] TO SC BT BT BT BT SA
                  4ccccd          2912         2879             0.19         1.01 +- 0.02         0.99 +- 0.02  [6 ] TO BT BT BT BT AB
                    4ccd          2264         2285             0.10         0.99 +- 0.02         1.01 +- 0.02  [4 ] TO BT BT AB
              cccc9ccccd          1402         2278           208.53         0.62 +- 0.02         1.62 +- 0.03  [10] TO BT BT BT BT DR BT BT BT BT
                 8cccc5d          2056         2050             0.01         1.00 +- 0.02         1.00 +- 0.02  [7 ] TO RE BT BT BT BT SA
                 86ccccd           961         1105            10.04         0.87 +- 0.03         1.15 +- 0.03  [7 ] TO BT BT BT BT SC SA
                 8cc6ccd          1035         1036             0.00         1.00 +- 0.03         1.00 +- 0.03  [7 ] TO BT BT SC BT BT SA
                 89ccccd           711            0           711.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR SA
                8cccc55d           601          641             1.29         0.94 +- 0.04         1.07 +- 0.04  [8 ] TO RE RE BT BT BT BT SA
                     45d           554          554             0.00         1.00 +- 0.04         1.00 +- 0.04  [3 ] TO RE AB
                     46d           539          435            11.10         1.24 +- 0.05         0.81 +- 0.04  [3 ] TO SC AB
                 8cc5ccd           485          468             0.30         1.04 +- 0.05         0.96 +- 0.04  [7 ] TO BT BT RE BT BT SA
              cccc6ccccd           461          333            20.63         1.38 +- 0.06         0.72 +- 0.04  [10] TO BT BT BT BT SC BT BT BT BT
              cccccc6ccd           346          339             0.07         1.02 +- 0.05         0.98 +- 0.05  [10] TO BT BT SC BT BT BT BT BT BT
                8cccc66d           280          267             0.31         1.05 +- 0.06         0.95 +- 0.06  [8 ] TO SC SC BT BT BT BT SA
                 4cccc6d           218          176             4.48         1.24 +- 0.08         0.81 +- 0.06  [7 ] TO SC BT BT BT BT AB
              5ccc9ccccd             0          207           207.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT RE
                 49ccccd           207          117            25.00         1.77 +- 0.12         0.57 +- 0.05  [7 ] TO BT BT BT BT DR AB
                              100000       100000        16.91 


After hookup of Optical Surfaces with test geometry, the top line 4% is fixed. This is with random direction source:: 

    [2016-11-01 17:22:57,862] p57567 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161101-1722 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-11-01 17:22:57,862] p57567 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161101-1722 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd         67138        67065             0.04         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d          8398         8284             0.78         1.01 +- 0.01         0.99 +- 0.01  [2 ] TO AB
                 8cccc6d          4574         4630             0.34         0.99 +- 0.01         1.01 +- 0.01  [7 ] TO SC BT BT BT BT SA
                  4ccccd          2912         2916             0.00         1.00 +- 0.02         1.00 +- 0.02  [6 ] TO BT BT BT BT AB
                    4ccd          2264         2374             2.61         0.95 +- 0.02         1.05 +- 0.02  [4 ] TO BT BT AB
              cccc9ccccd          1396         2267           207.11         0.62 +- 0.02         1.62 +- 0.03  [10] TO BT BT BT BT DR BT BT BT BT         ## truncation handling ???
                 8cccc5d          2030         2048             0.08         0.99 +- 0.02         1.01 +- 0.02  [7 ] TO RE BT BT BT BT SA
                 86ccccd           983         1086             5.13         0.91 +- 0.03         1.10 +- 0.03  [7 ] TO BT BT BT BT SC SA
                 8cc6ccd          1013         1058             0.98         0.96 +- 0.03         1.04 +- 0.03  [7 ] TO BT BT SC BT BT SA
                 89ccccd           725            0           725.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR SA                   ## post DR very different, ouch zero
                8cccc55d           612          635             0.42         0.96 +- 0.04         1.04 +- 0.04  [8 ] TO RE RE BT BT BT BT SA
                     45d           553          552             0.00         1.00 +- 0.04         1.00 +- 0.04  [3 ] TO RE AB
                 8cc5ccd           510          520             0.10         0.98 +- 0.04         1.02 +- 0.04  [7 ] TO BT BT RE BT BT SA
                     46d           511          445             4.56         1.15 +- 0.05         0.87 +- 0.04  [3 ] TO SC AB
              cccc6ccccd           472          348            18.75         1.36 +- 0.06         0.74 +- 0.04  [10] TO BT BT BT BT SC BT BT BT BT
              cccccc6ccd           355          333             0.70         1.07 +- 0.06         0.94 +- 0.05  [10] TO BT BT SC BT BT BT BT BT BT
                8cccc66d           278          254             1.08         1.09 +- 0.07         0.91 +- 0.06  [8 ] TO SC SC BT BT BT BT SA
                 49ccccd           221          115            33.44         1.92 +- 0.13         0.52 +- 0.05  [7 ] TO BT BT BT BT DR AB                  ## post DR very different
               8cccc555d           180          205             1.62         0.88 +- 0.07         1.14 +- 0.08  [9 ] TO RE RE RE BT BT BT BT SA
                 4cccc6d           200          202             0.01         0.99 +- 0.07         1.01 +- 0.07  [7 ] TO SC BT BT BT BT AB
                             100000       100000        17.81 



Two issues to investigate

* post DR very different
* truncation handling difference


seqhis
--------

::

    tconcentric.py 

    [2016-10-31 18:47:36,561] p24396 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161031-1837 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-10-31 18:47:36,562] p24396 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161031-1837 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
                  8ccccd         67105        69977            60.17         0.96 +- 0.00         1.04 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d          8398         8346             0.16         1.01 +- 0.01         0.99 +- 0.01  [2 ] TO AB
                 8cccc6d          4573         4732             2.72         0.97 +- 0.01         1.03 +- 0.02  [7 ] TO SC BT BT BT BT SA
                  4ccccd          2935         2876             0.60         1.02 +- 0.02         0.98 +- 0.02  [6 ] TO BT BT BT BT AB
                    4ccd          2264         2348             1.53         0.96 +- 0.02         1.04 +- 0.02  [4 ] TO BT BT AB
                 8cccc5d          2029         2102             1.29         0.97 +- 0.02         1.04 +- 0.02  [7 ] TO RE BT BT BT BT SA
         ##   cccc9ccccd          1389            0          1389.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT BT
                 8cc6ccd          1012         1137             7.27         0.89 +- 0.03         1.12 +- 0.03  [7 ] TO BT BT SC BT BT SA
                 86ccccd           992         1084             4.08         0.92 +- 0.03         1.09 +- 0.03  [7 ] TO BT BT BT BT SC SA
         ##      89ccccd           725            0           725.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR SA
                8cccc55d           612          602             0.08         1.02 +- 0.04         0.98 +- 0.04  [8 ] TO RE RE BT BT BT BT SA
                     45d           553          555             0.00         1.00 +- 0.04         1.00 +- 0.04  [3 ] TO RE AB
                     46d           511          494             0.29         1.03 +- 0.05         0.97 +- 0.04  [3 ] TO SC AB
                 8cc5ccd           510          482             0.79         1.06 +- 0.05         0.95 +- 0.04  [7 ] TO BT BT RE BT BT SA
              cccc6ccccd           474          372            12.30         1.27 +- 0.06         0.78 +- 0.04  [10] TO BT BT BT BT SC BT BT BT BT
              cccccc6ccd           355          308             3.33         1.15 +- 0.06         0.87 +- 0.05  [10] TO BT BT SC BT BT BT BT BT BT
                8cccc66d           278          238             3.10         1.17 +- 0.07         0.86 +- 0.06  [8 ] TO SC SC BT BT BT BT SA
                 49ccccd           222            0           222.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR AB
                 4cccc6d           201          210             0.20         0.96 +- 0.07         1.04 +- 0.07  [7 ] TO SC BT BT BT BT AB
                   4cc6d           196          195             0.00         1.01 +- 0.07         0.99 +- 0.07  [5 ] TO SC BT BT AB
                              100000       100000        40.19 


lack DR due to lack of complete Optical Surface ? with test geometry
-----------------------------------------------------------------------

::

    2016-10-31 20:46:48.434 INFO  [460591] [CMaterialTable::init@28] CMaterialTable::init  numOfMaterials 4 prefix /dd/Materials/
    2016-10-31 20:46:48.434 INFO  [460591] [CSkinSurfaceTable::init@25] CSkinSurfaceTable::init nsurf 36
        0               NearPoolCoverSurface               NearPoolCoverSurface lv NULL
        1      lvPmtHemiCathodeSensorSurface      lvPmtHemiCathodeSensorSurface lv NULL
        2    lvHeadonPmtCathodeSensorSurface    lvHeadonPmtCathodeSensorSurface lv NULL
        3                       RSOilSurface                       RSOilSurface lv NULL
        4                 AdCableTraySurface                 AdCableTraySurface lv NULL
        5                PmtMtTopRingSurface                PmtMtTopRingSurface lv NULL
        6               PmtMtBaseRingSurface               PmtMtBaseRingSurface lv NULL
        7                   PmtMtRib1Surface                   PmtMtRib1Surface lv NULL
        8                   PmtMtRib2Surface                   PmtMtRib2Surface lv NULL
        9                   PmtMtRib3Surface                   PmtMtRib3Surface lv NULL
       10                 LegInIWSTubSurface                 LegInIWSTubSurface lv NULL
       11                  TablePanelSurface                  TablePanelSurface lv NULL
       12                 SupportRib1Surface                 SupportRib1Surface lv NULL
       13                 SupportRib5Surface                 SupportRib5Surface lv NULL
       14                   SlopeRib1Surface                   SlopeRib1Surface lv NULL
       15                   SlopeRib5Surface                   SlopeRib5Surface lv NULL
       16            ADVertiCableTraySurface            ADVertiCableTraySurface lv NULL
       17           ShortParCableTraySurface           ShortParCableTraySurface lv NULL
       18              NearInnInPiperSurface              NearInnInPiperSurface lv NULL
       19             NearInnOutPiperSurface             NearInnOutPiperSurface lv NULL
       20                 LegInOWSTubSurface                 LegInOWSTubSurface lv NULL
       21                UnistrutRib6Surface                UnistrutRib6Surface lv NULL
       22                UnistrutRib7Surface                UnistrutRib7Surface lv NULL
       23                UnistrutRib3Surface                UnistrutRib3Surface lv NULL
       24                UnistrutRib5Surface                UnistrutRib5Surface lv NULL
       25                UnistrutRib4Surface                UnistrutRib4Surface lv NULL
       26                UnistrutRib1Surface                UnistrutRib1Surface lv NULL
       27                UnistrutRib2Surface                UnistrutRib2Surface lv NULL
       28                UnistrutRib8Surface                UnistrutRib8Surface lv NULL
       29                UnistrutRib9Surface                UnistrutRib9Surface lv NULL
       30           TopShortCableTraySurface           TopShortCableTraySurface lv NULL
       31          TopCornerCableTraySurface          TopCornerCableTraySurface lv NULL
       32              VertiCableTraySurface              VertiCableTraySurface lv NULL
       33              NearOutInPiperSurface              NearOutInPiperSurface lv NULL
       34             NearOutOutPiperSurface             NearOutOutPiperSurface lv NULL
       35                LegInDeadTubSurface                LegInDeadTubSurface lv NULL
    2016-10-31 20:46:48.435 INFO  [460591] [CBorderSurfaceTable::init@23] CBorderSurfaceTable::init nsurf 11
        0               NearDeadLinerSurface               NearDeadLinerSurface pv1 NULL  pv2 NULL 
        1                NearOWSLinerSurface                NearOWSLinerSurface pv1 NULL  pv2 NULL 
        2              NearIWSCurtainSurface              NearIWSCurtainSurface pv1 NULL  pv2 NULL 
        3               SSTWaterSurfaceNear1               SSTWaterSurfaceNear1 pv1 NULL  pv2 NULL 
        4                      SSTOilSurface                      SSTOilSurface pv1 NULL  pv2 NULL 




topline 8ccccd 4% Opticks deficit
-------------------------------------------------------

Complement check shows whole line of cfg4 zeros 

CFG4 modelling/recording mismatch, 

* it is just not doing... 9ccccd "TO BT BT BT BT DR xx"


::

    tconcentric.py --dbgseqhis 9ccccd

    [2016-10-31 19:00:08,913] p24442 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a concentric/torch/  1 :  20161031-1837 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    [2016-10-31 19:00:08,913] p24442 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b concentric/torch/ -1 :  20161031-1837 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
              seqhis_ana  1:concentric   -1:concentric           c2           ab           ba 
              cccc9ccccd          1389            0          1389.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT BT
                 89ccccd           725            0           725.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR SA
                 49ccccd           222            0           222.00         0.00 +- 0.00         0.00 +- 0.00  [7 ] TO BT BT BT BT DR AB
               4cc9ccccd            82            0            82.00         0.00 +- 0.00         0.00 +- 0.00  [9 ] TO BT BT BT BT DR BT BT AB
                869ccccd            78            0            78.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR SC SA
              c6cc9ccccd            75            0            75.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT SC BT
              c5cc9ccccd            34            0            34.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT RE BT
              ccc69ccccd            33            0            33.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR SC BT BT BT
              ccc99ccccd            19            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR DR BT BT BT
              55cc9ccccd            12            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT RE RE
              bccc9ccccd            12            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT BR
                899ccccd            11            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR DR SA
              45cc9ccccd             8            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT RE AB
                4c9ccccd             8            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR BT AB
                8b9ccccd             7            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR BR SA
              46cc9ccccd             6            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT SC AB
                469ccccd             5            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR SC AB
                499ccccd             5            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [8 ] TO BT BT BT BT DR DR AB
              4ccc9ccccd             5            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR BT BT BT AB
              4cc69ccccd             3            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO BT BT BT BT DR SC BT BT AB





::

    158 #ifdef USE_CUSTOM_BOUNDARY
    159 unsigned int OpBoundaryFlag(const DsG4OpBoundaryProcessStatus status)
    160 #else
    161 unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status)
    162 #endif
    163 {
    164     unsigned flag = 0 ;
    165     switch(status)
    166     {
    167         case FresnelRefraction:
    168         case SameMaterial:
    169                                flag=BOUNDARY_TRANSMIT;
    170                                break;
    171         case TotalInternalReflection:
    172         case       FresnelReflection:
    173                                flag=BOUNDARY_REFLECT;
    174                                break;
    175         case StepTooSmall:
    176                                flag=NAN_ABORT;
    177                                break;
    178         case Absorption:
    179                                flag=SURFACE_ABSORB ;
    180                                break;
    181         case Detection:
    182                                flag=SURFACE_DETECT ;
    183                                break;
    184         case SpikeReflection:
    185                                flag=SURFACE_SREFLECT ;
    186                                break;
    187         case LobeReflection:
    188         case LambertianReflection:
    189                                flag=SURFACE_DREFLECT ;
    190                                break;
    191         case Undefined:
    192         case BackScattering:
    193         case NotAtBoundary:
    194         case NoRINDEX:
    195 




Using dielectric_dielectric/groundfrontpainted for RSOilSurface would avoid DielectricMetal complications... 

::

     565         else if (type == dielectric_dielectric)
     566         {
     567             if ( theFinish == polishedfrontpainted || theFinish == groundfrontpainted )
     568             {
     569                 if( !G4BooleanRand(theReflectivity) )
     570                 {
     571                     DoAbsorption();
     572                 }
     573                 else
     574                 {
     575                     if ( theFinish == groundfrontpainted ) theStatus = LambertianReflection;
     576                     DoReflection();
     577                 }
     578             }
     579             else
     580             {
     581                 DielectricDielectric();
     582             }
     583         }







::

    simon:opticks blyth$ op --surf 8
    === op-cmdline-binary-match : finds 1st argument with associated binary : --surf
    224 -rwxr-xr-x  1 blyth  staff  112772 Oct 31 17:29 /usr/local/opticks/lib/GSurfaceLibTest
    proceeding : /usr/local/opticks/lib/GSurfaceLibTest 8
    2016-10-31 19:12:46.462 INFO  [424703] [GSurfaceLib::Summary@137] GSurfaceLib::dump NumSurfaces 48 NumFloat4 2
    2016-10-31 19:12:46.462 INFO  [424703] [GSurfaceLib::dump@654]  (index,type,finish,value) 
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]                NearPoolCoverSurface (  0,  0,  3,100)  (  0)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]                NearDeadLinerSurface (  1,  0,  3, 20)  (  1)               dielectric_metal                        ground value 20
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]                 NearOWSLinerSurface (  2,  0,  3, 20)  (  2)               dielectric_metal                        ground value 20
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]               NearIWSCurtainSurface (  3,  0,  3, 20)  (  3)               dielectric_metal                        ground value 20
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]                SSTWaterSurfaceNear1 (  4,  0,  3,100)  (  4)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]                       SSTOilSurface (  5,  0,  3,100)  (  5)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.462 WARN  [424703] [GSurfaceLib::dump@661]       lvPmtHemiCathodeSensorSurface (  6,  0,  3,100)  (  6)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]     lvHeadonPmtCathodeSensorSurface (  7,  0,  3,100)  (  7)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                        RSOilSurface (  8,  0,  3,100)  (  8)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    ESRAirSurfaceTop (  9,  0,  0,  0)  (  9)               dielectric_metal                      polished value 0
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    ESRAirSurfaceBot ( 10,  0,  0,  0)  ( 10)               dielectric_metal                      polished value 0
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                  AdCableTraySurface ( 11,  0,  3,100)  ( 11)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                SSTWaterSurfaceNear2 ( 12,  0,  3,100)  ( 12)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                 PmtMtTopRingSurface ( 13,  0,  3,100)  ( 13)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                PmtMtBaseRingSurface ( 14,  0,  3,100)  ( 14)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    PmtMtRib1Surface ( 15,  0,  3,100)  ( 15)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    PmtMtRib2Surface ( 16,  0,  3,100)  ( 16)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    PmtMtRib3Surface ( 17,  0,  3,100)  ( 17)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                  LegInIWSTubSurface ( 18,  0,  3,100)  ( 18)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                   TablePanelSurface ( 19,  0,  3,100)  ( 19)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                  SupportRib1Surface ( 20,  0,  3,100)  ( 20)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                  SupportRib5Surface ( 21,  0,  3,100)  ( 21)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    SlopeRib1Surface ( 22,  0,  3,100)  ( 22)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]                    SlopeRib5Surface ( 23,  0,  3,100)  ( 23)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]             ADVertiCableTraySurface ( 24,  0,  3,100)  ( 24)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]            ShortParCableTraySurface ( 25,  0,  3,100)  ( 25)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.463 WARN  [424703] [GSurfaceLib::dump@661]               NearInnInPiperSurface ( 26,  0,  3,100)  ( 26)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]              NearInnOutPiperSurface ( 27,  0,  3,100)  ( 27)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                  LegInOWSTubSurface ( 28,  0,  3,100)  ( 28)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib6Surface ( 29,  0,  3,100)  ( 29)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib7Surface ( 30,  0,  3,100)  ( 30)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib3Surface ( 31,  0,  3,100)  ( 31)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib5Surface ( 32,  0,  3,100)  ( 32)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib4Surface ( 33,  0,  3,100)  ( 33)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib1Surface ( 34,  0,  3,100)  ( 34)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib2Surface ( 35,  0,  3,100)  ( 35)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib8Surface ( 36,  0,  3,100)  ( 36)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 UnistrutRib9Surface ( 37,  0,  3,100)  ( 37)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]            TopShortCableTraySurface ( 38,  0,  3,100)  ( 38)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]           TopCornerCableTraySurface ( 39,  0,  3,100)  ( 39)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]               VertiCableTraySurface ( 40,  0,  3,100)  ( 40)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]               NearOutInPiperSurface ( 41,  0,  3,100)  ( 41)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]              NearOutOutPiperSurface ( 42,  0,  3,100)  ( 42)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                 LegInDeadTubSurface ( 43,  0,  3,100)  ( 43)               dielectric_metal                        ground value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                perfectDetectSurface ( 44,  1,  1,100)  ( 44)          dielectric_dielectric          polishedfrontpainted value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]                perfectAbsorbSurface ( 45,  1,  1,100)  ( 45)          dielectric_dielectric          polishedfrontpainted value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]              perfectSpecularSurface ( 46,  1,  1,100)  ( 46)          dielectric_dielectric          polishedfrontpainted value 100
    2016-10-31 19:12:46.464 WARN  [424703] [GSurfaceLib::dump@661]               perfectDiffuseSurface ( 47,  1,  1,100)  ( 47)          dielectric_dielectric          polishedfrontpainted value 100

    2016-10-31 19:12:46.464 INFO  [424703] [GSurfaceLib::dump@720]  (  8,  0,  3,100) GPropertyMap<T>::  8        surface s: GOpticalSurface  type 0 model 1 finish 3 value     1                  RSOilSurface k:detect absorb reflect_specular reflect_diffuse extra_x extra_y extra_z extra_w RSOilSurface
                  domain              detect              absorb    reflect_specular     reflect_diffuse             extra_x
                      60                   0               0.827                   0               0.173                  -1
                      80                   0            0.827015                   0            0.172985                  -1
                     100                   0             0.85649                   0             0.14351                  -1
                     120                   0            0.885965                   0            0.114035                  -1
                     140                   0            0.897743                   0            0.102257                  -1
                     160                   0            0.909501                   0           0.0904994                  -1
                     180                   0            0.921258                   0           0.0787423                  -1
                     200                   0            0.933007                   0           0.0669933                  -1
                     220                   0            0.938282                   0           0.0617179                  -1
                     240                   0            0.943557                   0           0.0564426                  -1
                     260                   0            0.947648                   0           0.0523518                  -1
                     280                   0             0.95055                   0           0.0494499                  -1
                     300                   0            0.953451                   0           0.0465491                  -1
                     320                   0            0.954789                   0           0.0452105                  -1
                     340                   0            0.956128                   0            0.043872                  -1
                     360                   0            0.957098                   0           0.0429022                  -1
                     380                   0            0.957696                   0           0.0423041                  -1
                     400                   0            0.958294                   0           0.0417061                  -1
                     420                   0            0.958841                   0            0.041159                  -1
                     440                   0            0.959313                   0           0.0406869                  -1
                     460                   0             0.95969                   0           0.0403102                  -1
                     480                   0             0.95997                   0           0.0400297                  -1
                     500                   0             0.96025                   0           0.0397498                  -1
                     520                   0             0.96032                   0           0.0396799                  -1
                     540                   0             0.96039                   0             0.03961                  -1
                     560                   0             0.96046                   0           0.0395402                  -1
                     580                   0             0.96053                   0           0.0394703                  -1
                     600                   0              0.9606                   0           0.0394004                  -1
                     620                   0             0.96062                   0           0.0393801                  -1
                     640                   0             0.96064                   0           0.0393601                  -1
                     660                   0             0.96066                   0           0.0393401                  -1
                     680                   0             0.96068                   0           0.0393201                  -1
                     700                   0              0.9607                   0           0.0393001                  -1
                     720                   0              0.9607                   0              0.0393                  -1
                     740                   0              0.9607                   0              0.0393                  -1
                     760                   0              0.9607                   0              0.0393                  -1
                     780                   0              0.9607                   0              0.0393                  -1
                     800                   0              0.9607                   0              0.0393                  -1
                     820                   0              0.9607                   0              0.0393                  -1




