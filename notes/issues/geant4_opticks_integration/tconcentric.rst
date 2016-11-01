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




