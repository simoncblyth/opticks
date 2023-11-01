G4CXTest_raindrop_shakedown
=============================

::

    ~/opticks/g4cx/tests/G4CXTest_raindrop.sh


A : Opticks is scattering and absorbing 
B : Geant4 is not scattering and absorbing  

* it is as if different physics OR different material properties 

* U4Physics process switches are not honoured by Opticks : 
  but I didnt switch anything off ... just not expecting 
  scattering and absorption  

* but I didnt input scattering lengths and absorption lengths in this simple test ? 
  only RINDEX ? 


* Q: Where is Opticks getting the material properties ? 

* HMM: this may be Opticks standardization defaults ?

  * recall that when no property is available Opticks adopts 
    some large values for scattering and absorption length

  * the issue only becomes apparent with large (1M level) stats 

  * TODO : confirm this and increase the defaults hugely 
    to match Geant4 in anything but insane stats


Default scattering and absorption length of 1e6 mm is too small
------------------------------------------------------------------

::

    epsilon:standard blyth$ GEOM std
    cd /Users/blyth/.opticks/GEOM/RaindropRockAirWater/CSGFoundry/SSim/stree/standard

    epsilon:standard blyth$ t f 
    f () 
    { 
        ~/np/f.sh;
        : ~/.bash_profile
    }
    epsilon:standard blyth$ f 
    f

    ...

    In [1]: f.mat.shape                                                                                                                   
    Out[1]: (3, 2, 761, 4)

    In [2]: f.mat[:,0]                                                                                                                    
    Out[2]: 
    array([[[      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            ...,
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ],
            [      1.333, 1000000.   , 1000000.   ,       0.   ]],

           [[      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            ...,
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ]],

           [[      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            ...,
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ],
            [      1.   , 1000000.   , 1000000.   ,       0.   ]]])

    In [3]:           





Steps to debug
----------------

1. persist GEOM to file and peruse 
2. PIDX debug 


Issue 2 : lack of metadata re GPU in use etc..
-----------------------------------------------

Issue 1 : very different A/B histories as if different material props OR physics
-----------------------------------------------------------------------------------

::

    ~/opticks/g4cx/tests/G4CXTest_raindrop.sh

    ...

    ab.qcf[:40]
    QCF qcf :  
    a.q 1000000 b.q 1000000 lim slice(None, None, None) 
    c2sum :   298.6566 c2n :    15.0000 c2per:    19.9104  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  298.66/15:19.910 (30)

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:40]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT BT SA                  ' ' 0' '879742 879870' ' 0.0093' '     0      0']
     [' 1' 'TO BR SA                     ' ' 1' ' 66755  67104' ' 0.9099' '    62     53']
     [' 2' 'TO BT BR BT SA               ' ' 2' ' 45134  44838' ' 0.9738' '     1      2']
     [' 3' 'TO BT BR BR BT SA            ' ' 3' '  5472   5580' ' 1.0554' '   345    344']
     [' 4' 'TO BT BR BR BR BT SA         ' ' 4' '  1360   1494' ' 6.2915' '  4471    358']
     [' 5' 'TO BT BR BR BR BR BT SA      ' ' 5' '   536    557' ' 0.4035' '     3   1106']
     [' 6' 'TO BT BR BR BR BR BR BT SA   ' ' 6' '   240    209' ' 2.1403' '  5689   5331']
     [' 7' 'TO BT BR BR BR BR BR BR BR BR' ' 7' '   173    147' ' 2.1125' '  3145    896']
     [' 8' 'TO BT BR BR BR BR BR BR BT SA' ' 8' '   115    123' ' 0.2689' '  3602   1783']
     [' 9' 'TO BT BR BR BR BR BR BR BR BT' ' 9' '    99     78' ' 2.4915' ' 13398   1424']
     ['10' 'TO BT AB                     ' '10' '    97      0' '97.0000' ' 11587     -1']
     ['11' 'TO BT SC BT SA               ' '11' '    68      0' '68.0000' '  9989     -1']
     ['12' 'TO AB                        ' '12' '    45      0' '45.0000' ' 33262     -1']
     ['13' 'TO BT BT AB                  ' '13' '    38      0' '38.0000' ' 11664     -1']
     ['14' 'TO BT BT SC SA               ' '14' '    34      0' '34.0000' ' 73257     -1']
     ['15' 'TO SC SA                     ' '15' '    29      0' ' 0.0000' '  8602     -1']
     ['16' 'TO BT SC BR BR BR BR BR BR BR' '16' '    10      0' ' 0.0000' '109353     -1']
     ['17' 'TO SC BT BT SA               ' '17' '     9      0' ' 0.0000' '171775     -1']
     ['18' 'TO BT BT SC BT BT SA         ' '18' '     8      0' ' 0.0000' '204411     -1']
     ['19' 'TO BT BR AB                  ' '19' '     7      0' ' 0.0000' ' 92992     -1']
     ['20' 'TO BR AB                     ' '20' '     6      0' ' 0.0000' '184076     -1']
     ['21' 'TO BT SC BR BT SA            ' '21' '     5      0' ' 0.0000' ' 10048     -1']
     ['22' 'TO BT BR BT SC SA            ' '22' '     4      0' ' 0.0000' ' 76929     -1']



