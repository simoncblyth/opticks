SURFACE_DREFLECT (DR) diffuse reflection
==========================================

issue : now resolved
----------------------

During tconcentric checking found that polarization
distribs following DR were discrepant.
Investigation showed cause to simply be a different
implementation between CFG4 and Opticks. Addition
and use of `propagate_at_diffuse_reflector_geant4_style`
resolved the issue.



CFG4 SURFACE_DREFLECT from LobeReflection or LambertianReflection
-------------------------------------------------------------------

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
    ...
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


Cause of discrepant reflection, different imps
--------------------------------------------------

::

    288 inline
    289 void DsG4OpBoundaryProcess::DoReflection()
    290 {
    291         if ( theStatus == LambertianReflection ) {
    292 
    293           NewMomentum = G4LambertianRand(theGlobalNormal);
    294           theFacetNormal = (NewMomentum - OldMomentum).unit();
    295 
    296         }
    297         else if ( theFinish == ground ) {
    298 
    299           theStatus = LobeReflection;
    300           theFacetNormal = GetFacetNormal(OldMomentum,theGlobalNormal);
    301           G4double PdotN = OldMomentum * theFacetNormal;
    302           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    303 
    304         }
    305         else {
    306 
    307           theStatus = SpikeReflection;
    308           theFacetNormal = theGlobalNormal;
    309           G4double PdotN = OldMomentum * theFacetNormal;
    310           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    311 
    312         }
    313         G4double EdotN = OldPolarization * theFacetNormal;
    314         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    315 }


::

    simon:geant4_10_02_p01 blyth$ find source -name '*.*' -exec grep -H G4LambertianRand {} \;
    source/global/HEPRandom/include/G4RandomTools.hh:inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
    source/processes/optical/include/G4OpBoundaryProcess.hh:          NewMomentum = G4LambertianRand(theGlobalNormal);

    055 inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
     56 {
     57   G4ThreeVector vect;
     58   G4double ndotv;
     59   G4int count=0;
     60   const G4int max_trials = 1024;
     61 
     62   do
     63   {
     64     ++count;
     65     vect = G4RandomDirection();
     66     ndotv = normal * vect;
     67 
     68     if (ndotv < 0.0)
     69     {
     70       vect = -vect;
     71       ndotv = -ndotv;
     72     }
     73 
     74   } while (!(G4UniformRand() < ndotv) && (count < max_trials));
     75 
     76   return vect;
     77 }



Reimplement Opticks diffuse reflection
------------------------------------------

::

    434 __device__ void propagate_at_diffuse_reflector(Photon &p, State &s, curandState &rng)
    435 {
    436     float ndotv;
    437     do {
    438         p.direction = uniform_sphere(&rng);
    439         ndotv = dot(p.direction, s.surface_normal);
    440         if (ndotv < 0.0f)
    441         {
    442             p.direction = -p.direction;
    443             ndotv = -ndotv;
    444         }
    445     } while (! (curand_uniform(&rng) < ndotv) );
    446 
    447     p.polarization = normalize( cross(uniform_sphere(&rng), p.direction));
    449     p.flags.i.x = 0 ;  // no-boundary-yet for new direction
    450 }


    451 __device__ void propagate_at_diffuse_reflector_geant4_style(Photon &p, State &s, curandState &rng)
    452 {
    453 
    454     float3 old_direction = p.direction ;
    455 
    456     float ndotv;
    457     do {
    458         p.direction = uniform_sphere(&rng);
    459         ndotv = dot(p.direction, s.surface_normal);
    460         if (ndotv < 0.0f)
    461         {
    462             p.direction = -p.direction;
    463             ndotv = -ndotv;
    464         }
    465     } while (! (curand_uniform(&rng) < ndotv) );
    466 
    467 
    468     float3 facet_normal = normalize( p.direction - old_direction ) ;
    469 
    470     float normal_coefficient = dot(p.polarization, facet_normal);  // EdotN
    471 
    472     p.polarization = -p.polarization + 2.f*normal_coefficient*facet_normal ;
    473 
    474     p.flags.i.x = 0 ;  // no-boundary-yet for new direction
    475 }
    476 



SURFACE_DREFLECT discrepant polarization b,c
----------------------------------------------

::

    8              89ccccd          7605         7694             0.52        0.988 +- 0.011        1.012 +- 0.012  [7 ] TO BT BT BT BT DR SA


    tconcentric-d

    In [1]: cf.ss[0]
    Out[1]: CF(1,torch,concentric,['TO BT BT BT BT DR SA']) 

    In [2]: scf = cf.ss[0]

    In [3]: a, b = scf.rpol()

    In [32]: a.shape,b.shape
    Out[32]: ((7605, 7, 3), (7694, 7, 3))

    In [9]: a[0]   # pol at last two points same for a and b 
    A([    [ 0.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  0.    ],
           [-0.7953, -0.3071, -0.5276],
           [-0.7953, -0.3071, -0.5276]], dtype=float32)


    In [33]: vnorm(b[:,-1]).min(),vnorm(b[:,-1]).max(),vnorm(a[:,-1]).min(),vnorm(a[:,-1]).max()  ## normalization ok
    Out[33]: 
    A(0.9940925240516663, dtype=float32),
    A(1.0060884952545166, dtype=float32),
    A(0.993811845779419, dtype=float32),
    A(1.0064274072647095, dtype=float32))

    plt.hist(a[:,-1,1], bins=100, histtype="step")  ## lots of compression bin moire, but issue still apparent 
    plt.hist(b[:,-1,1], bins=100, histtype="step")



NumPy DoReflection calc
------------------------------------

::

    tconcentric-d   # loads the evt 

    In [70]: oa, ob = scf.rdir(4,5)   # old momdir, before DR
    In [73]: la, lb = scf.rpol_(4)    # old pol 

    In [40]: da, db = scf.rdir(5,6)   # direction after DR
    In [44]: pa, pb = scf.rpol_(5)    # pol after DR

    In [58]: cb = costheta_(db, pb )   # matching precision bump around zero, so they stay transverse
    In [59]: ca = costheta_(da, pa )

    In [81]: norm_ = lambda a:a/np.repeat(vnorm(a), 3).reshape(-1,3)
    In [82]: na = norm_(da-oa)    #   theFacetNormal = (NewMomentum - OldMomentum).unit();   midway between old and new directions
    In [84]: nb = norm_(db-ob)


    In [90]: ea = np.sum(la * na, axis=1)     # G4double EdotN = OldPolarization * theFacetNormal;
    In [92]: eb = np.sum(lb * nb, axis=1)

    In [91]: ea
    Out[91]: 
    A([ 0.0282, -0.0403, -0.13  , ..., -0.1157,  0.2941,  0.4657])


    In [103]: qb = -lb + np.repeat(2*eb, 3).reshape(-1,3)*nb      # NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;

    In [104]: qb                        # numpy calc of pol after DR (using highly compressed dir and pol) 
    Out[104]: 
    A()sliced
    A([[ 0.4907, -0.7914,  0.3644],
           [ 0.4491, -0.8669, -0.2163],
           [ 0.6693, -0.6506,  0.3589],
           ..., 
           [-0.0898, -0.9939, -0.0637],
           [-0.3176, -0.9245,  0.2106],
           [ 0.9044, -0.3842,  0.1854]])

    In [105]: pb        # pol after DR from CFG4
    Out[105]: 
    A()sliced
    A([[ 0.4882, -0.7953,  0.3622],
           [ 0.4488, -0.8661, -0.2126],
           [ 0.6693, -0.6535,  0.3622],
           ..., 
           [-0.0866, -0.9921, -0.063 ],
           [-0.315 , -0.9213,  0.2126],
           [ 0.9055, -0.3858,  0.189 ]], dtype=float32)


    In [109]: ((qb - pb).min(),(qb - pb).max())      ## expected level of agreement given the compression
    Out[109]: 
    (A()sliced
    A(-0.003979883117690708), A()sliced
    A(0.004024292009927266))



chi2 after fix, notably the pflags decreased from 1.21 to 1.07
------------------------------------------------------------------

::

    simon:geant4_opticks_integration blyth$ tconcentric.py 
    /Users/blyth/opticks/ana/tconcentric.py
    [2016-11-08 13:11:16,203] p73931 {/Users/blyth/opticks/ana/tconcentric.py:208} INFO - tag 1 src torch det concentric c2max 2.0 ipython False 
    [2016-11-08 13:11:16,813] p73931 {/Users/blyth/opticks/ana/evt.py:400} INFO - pflags2(=seq2msk(seqhis)) and pflags  match
    [2016-11-08 13:11:17,103] p73931 {/Users/blyth/opticks/ana/evt.py:474} WARNING - _init_selection with psel None : resetting selection to original 
    [2016-11-08 13:11:19,810] p73931 {/Users/blyth/opticks/ana/evt.py:400} INFO - pflags2(=seq2msk(seqhis)) and pflags  match
    [2016-11-08 13:11:20,100] p73931 {/Users/blyth/opticks/ana/evt.py:474} WARNING - _init_selection with psel None : resetting selection to original 
    CF a concentric/torch/  1 :  20161108-1253 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    CF b concentric/torch/ -1 :  20161108-1253 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    [2016-11-08 13:11:22,222] p73931 {/Users/blyth/opticks/ana/seq.py:410} INFO - compare dbgseq 0 dbgmsk 0 
    .                seqhis_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       363.45/354 =  1.03  (pval:0.353 prob:0.647)  
       0               8ccccd        669843       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [6 ] TO BT BT BT BT SA
       1                   4d         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] TO AB
       2              8cccc6d         45490        45054             2.10        1.010 +- 0.005        0.990 +- 0.005  [7 ] TO SC BT BT BT BT SA
       3               4ccccd         28955        28649             1.63        1.011 +- 0.006        0.989 +- 0.006  [6 ] TO BT BT BT BT AB
       4                 4ccd         23187        23254             0.10        0.997 +- 0.007        1.003 +- 0.007  [4 ] TO BT BT AB
       5              8cccc5d         20239        19946             2.14        1.015 +- 0.007        0.986 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              86ccccd         10176        10396             2.35        0.979 +- 0.010        1.022 +- 0.010  [7 ] TO BT BT BT BT SC SA
       7              8cc6ccd         10214        10304             0.39        0.991 +- 0.010        1.009 +- 0.010  [7 ] TO BT BT SC BT BT SA
       8              89ccccd          7540         7694             1.56        0.980 +- 0.011        1.020 +- 0.012  [7 ] TO BT BT BT BT DR SA
       9             8cccc55d          5970         5814             2.07        1.027 +- 0.013        0.974 +- 0.013  [8 ] TO RE RE BT BT BT BT SA
      10                  45d          5780         5658             1.30        1.022 +- 0.013        0.979 +- 0.013  [3 ] TO RE AB
      11      8cccccccc9ccccd          5339         5367             0.07        0.995 +- 0.014        1.005 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      12              8cc5ccd          5113         4868             6.01        1.050 +- 0.015        0.952 +- 0.014  [7 ] TO BT BT RE BT BT SA
      13                  46d          4797         4815             0.03        0.996 +- 0.014        1.004 +- 0.014  [3 ] TO SC AB
      14          8cccc9ccccd          4494         4420             0.61        1.017 +- 0.015        0.984 +- 0.015  [11] TO BT BT BT BT DR BT BT BT BT SA
      15          8cccccc6ccd          3317         3333             0.04        0.995 +- 0.017        1.005 +- 0.017  [11] TO BT BT SC BT BT BT BT BT BT SA
      16             8cccc66d          2670         2734             0.76        0.977 +- 0.019        1.024 +- 0.020  [8 ] TO SC SC BT BT BT BT SA
      17              49ccccd          2432         2472             0.33        0.984 +- 0.020        1.016 +- 0.020  [7 ] TO BT BT BT BT DR AB
      18              4cccc6d          2043         2042             0.00        1.000 +- 0.022        1.000 +- 0.022  [7 ] TO SC BT BT BT BT AB
      19            8cccc555d          1819         1762             0.91        1.032 +- 0.024        0.969 +- 0.023  [9 ] TO RE RE RE BT BT BT BT SA
    .                               1000000      1000000       363.45/354 =  1.03  (pval:0.353 prob:0.647)  
    [2016-11-08 13:11:22,362] p73931 {/Users/blyth/opticks/ana/seq.py:410} INFO - compare dbgseq 0 dbgmsk 0 
    .                pflags_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000        44.95/42 =  1.07  (pval:0.349 prob:0.651)  
       0                 1880        669843       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [3 ] TO|BT|SA
       1                 1008         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] TO|AB
       2                 18a0         79906        79772             0.11        1.002 +- 0.004        0.998 +- 0.004  [4 ] TO|BT|SA|SC
       3                 1808         54172        53852             0.95        1.006 +- 0.004        0.994 +- 0.004  [3 ] TO|BT|AB
       4                 1890         38518        37832             6.16        1.018 +- 0.005        0.982 +- 0.005  [4 ] TO|BT|SA|RE
       5                 1980         17710        17843             0.50        0.993 +- 0.007        1.008 +- 0.008  [4 ] TO|BT|DR|SA
       6                 1828          8788         9013             2.84        0.975 +- 0.010        1.026 +- 0.011  [4 ] TO|BT|SC|AB
       7                 1018          8204         8002             2.52        1.025 +- 0.011        0.975 +- 0.011  [3 ] TO|RE|AB
       8                 18b0          7901         7879             0.03        1.003 +- 0.011        0.997 +- 0.011  [5 ] TO|BT|SA|SC|RE
       9                 1818          6024         5941             0.58        1.014 +- 0.013        0.986 +- 0.013  [4 ] TO|BT|RE|AB
      10                 1908          5531         5463             0.42        1.012 +- 0.014        0.988 +- 0.013  [4 ] TO|BT|DR|AB
      11                 1028          5089         5153             0.40        0.988 +- 0.014        1.013 +- 0.014  [3 ] TO|SC|AB
      12                 19a0          4931         4928             0.00        1.001 +- 0.014        0.999 +- 0.014  [5 ] TO|BT|DR|SA|SC
      13                 1990          1481         1541             1.19        0.961 +- 0.025        1.041 +- 0.027  [5 ] TO|BT|DR|SA|RE
      14                 1838          1540         1535             0.01        1.003 +- 0.026        0.997 +- 0.025  [5 ] TO|BT|SC|RE|AB
      15                 1928          1056         1085             0.39        0.973 +- 0.030        1.027 +- 0.031  [5 ] TO|BT|DR|SC|AB
      16                 1920           789          759             0.58        1.040 +- 0.037        0.962 +- 0.035  [4 ] TO|BT|DR|SC
      17                 1038           770          776             0.02        0.992 +- 0.036        1.008 +- 0.036  [4 ] TO|SC|RE|AB
      18                 1918           630          609             0.36        1.034 +- 0.041        0.967 +- 0.039  [5 ] TO|BT|DR|RE|AB
      19                 1910           491          410             7.28        1.198 +- 0.054        0.835 +- 0.041  [4 ] TO|BT|DR|RE
    .                               1000000      1000000        44.95/42 =  1.07  (pval:0.349 prob:0.651)  
    [2016-11-08 13:11:22,392] p73931 {/Users/blyth/opticks/ana/seq.py:410} INFO - compare dbgseq 0 dbgmsk 0 
    .                seqmat_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       222.91/230 =  0.97  (pval:0.619 prob:0.381)  
       0               343231        669845       671267             1.51        0.998 +- 0.001        1.002 +- 0.001  [6 ] Gd Ac LS Ac MO Ac
       1                   11         83950        83637             0.58        1.004 +- 0.003        0.996 +- 0.003  [2 ] Gd Gd
       2              3432311         65732        65001             4.09        1.011 +- 0.004        0.989 +- 0.004  [7 ] Gd Gd Ac LS Ac MO Ac
       3               443231         28955        28649             1.63        1.011 +- 0.006        0.989 +- 0.006  [6 ] Gd Ac LS Ac MO MO
       4                 2231         23188        23254             0.09        0.997 +- 0.007        1.003 +- 0.007  [4 ] Gd Ac LS LS
       5              3443231         17716        18090             3.91        0.979 +- 0.007        1.021 +- 0.008  [7 ] Gd Ac LS Ac MO MO Ac
       6              3432231         15327        15172             0.79        1.010 +- 0.008        0.990 +- 0.008  [7 ] Gd Ac LS LS Ac MO Ac
       7             34323111         10934        10826             0.54        1.010 +- 0.010        0.990 +- 0.010  [8 ] Gd Gd Gd Ac LS Ac MO Ac
       8                  111         10577        10474             0.50        1.010 +- 0.010        0.990 +- 0.010  [3 ] Gd Gd Gd
       9      343231323443231          6955         7001             0.15        0.993 +- 0.012        1.007 +- 0.012  [15] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Ac
      10          34323443231          6038         5954             0.59        1.014 +- 0.013        0.986 +- 0.013  [11] Gd Ac LS Ac MO MO Ac LS Ac MO Ac
      11          34323132231          4422         4532             1.35        0.976 +- 0.015        1.025 +- 0.015  [11] Gd Ac LS LS Ac Gd Ac LS Ac MO Ac
      12              4443231          3160         3272             1.95        0.966 +- 0.017        1.035 +- 0.018  [7 ] Gd Ac LS Ac MO MO MO
      13              4432311          3008         3002             0.01        1.002 +- 0.018        0.998 +- 0.018  [7 ] Gd Gd Ac LS Ac MO MO
      14            343231111          2859         2860             0.00        1.000 +- 0.019        1.000 +- 0.019  [9 ] Gd Gd Gd Gd Ac LS Ac MO Ac
      15                22311          2791         2754             0.25        1.013 +- 0.019        0.987 +- 0.019  [5 ] Gd Gd Ac LS LS
      16                 1111          2446         2437             0.02        1.004 +- 0.020        0.996 +- 0.020  [4 ] Gd Gd Gd Gd
      17             34322311          1999         1869             4.37        1.070 +- 0.024        0.935 +- 0.022  [8 ] Gd Gd Ac LS LS Ac MO Ac
      18             34322231          1844         1872             0.21        0.985 +- 0.023        1.015 +- 0.023  [8 ] Gd Ac LS LS LS Ac MO Ac
      19                22231          1790         1825             0.34        0.981 +- 0.023        1.020 +- 0.024  [5 ] Gd Ac LS LS LS
    .                               1000000      1000000       222.91/230 =  0.97  (pval:0.619 prob:0.381)  
    [2016-11-08 13:11:22,450] p73931 {/Users/blyth/opticks/ana/evt.py:750} WARNING - missing a_ana hflags_ana 
    [2016-11-08 13:11:22,450] p73931 {/Users/blyth/opticks/ana/tconcentric.py:213} INFO - early exit as non-interactive
    simon:geant4_opticks_integration blyth$ 

