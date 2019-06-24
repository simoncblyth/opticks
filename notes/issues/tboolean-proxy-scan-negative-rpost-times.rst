tboolean-proxy-scan-negative-rpost-times
=============================================

Context
-----------

* :doc:`tboolean-proxy-scan`


Command shortcuts
---------------------

::

    lv(){ echo 21 ; }
    # default geometry LV index to test 

    ts(){  LV=${1:-$(lv)} tboolean.sh --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $* ; } 
    # **simulate** : aligned bi-simulation creating OK+G4 events 

    tv(){  LV=${1:-$(lv)} tboolean.sh --load $* ; } 
    # **visualize** : load events and visualize the propagation

    tv4(){  LV=${1:-$(lv)} tboolean.sh --load --vizg4 $* ; } 
    # **visualize** the geant4 propagation 

    ta(){  tboolean-;LV=${1:-$(lv)} tboolean-proxy-ip ; } 
    # **analyse** : load events and analyse the propagation


rejigged shortcuts moving the above options within tboolean-lv
------------------------------------------------------------------

::

    [blyth@localhost ana]$ t opticks-tboolean-shortcuts
    opticks-tboolean-shortcuts is a function
    opticks-tboolean-shortcuts () 
    { 
        : default geometry LV index or tboolean-geomname eg "box" "sphere" etc..;
        function lv () 
        { 
            echo 21
        };
        : **simulate** : aligned bi-simulation creating OK+G4 events;
        function ts () 
        { 
            LV=${1:-$(lv)} tboolean.sh $*
        };
        : **visualize** : load events and visualize the propagation;
        function tv () 
        { 
            LV=${1:-$(lv)} tboolean.sh --load $*
        };
        : **visualize** the geant4 propagation;
        function tv4 () 
        { 
            LV=${1:-$(lv)} tboolean.sh --load --vizg4 $*
        };
        : **analyse** : load events and analyse the propagation;
        function ta () 
        { 
            LV=${1:-$(lv)} tboolean.sh --ip
        }
    }





ISSUE : -ve rpost times from too small time domain
---------------------------------------------------------


LV:10 
~~~~~~~~~~~~~~

ta 10::

    In [28]: a.rpost().shape
    Out[28]: (23, 5, 4)

    In [26]: ab.a.rpost()       # negative time at the top 
    Out[26]: 
    A()sliced
    A([[[    39.5525,   -188.9732, -71998.8026,      0.    ],
        [    39.5525,   -188.9732,  -2500.5993,    231.8218],
        [    39.5525,   -188.9732,   1500.7991,    245.1671],
        [    39.5525,   -188.9732,   2500.5993,    251.2319],
        [    39.5525,   -188.9732,  72001.    ,   -480.0213]],

       [[  -239.5126,    -92.2893, -71998.8026,      0.    ],
        [  -239.5126,    -92.2893,  -2500.5993,    231.8218],
        [  -239.5126,    -92.2893,   1500.7991,    245.1671],
        [  -239.5126,    -92.2893,   2500.5993,    251.2319],
        [  -239.5126,    -92.2893,  72001.    ,   -480.0213]],


    In [29]: ab.a.fdom
    Out[29]: 
    A(torch,1,tboolean-proxy-10)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[    0.    ,     0.    ,     0.    , 72001.    ]],

       [[    0.    ,   480.0067,   480.0067,     0.    ]],

       [[   60.    ,   820.    ,    20.    ,   760.    ]]], dtype=float32)


Not fitting in short spits out SHRT_MIN -32767 as the compressed time
becoming -480.0 the negated timemax : which stands out like a sore thumb.  

::

   TMAX=500 ts 10    ## it was close 



    In [1]: ab.sel = "TO BT BT BT SA"
    [2019-06-21 23:30:18,744] p83910 {evt.py    :876} WARNING  - _init_selection EMPTY nsel 0 len(psel) 10000 

    In [2]: a.rpost().shape
    Out[2]: (23, 5, 4)

    In [3]: ab.a.rpost()
    Out[3]: 
    A()sliced
    A([[[    39.5525,   -188.9732, -71998.8026,      0.    ],
        [    39.5525,   -188.9732,  -2500.5993,    231.8339],
        [    39.5525,   -188.9732,   1500.7991,    245.1704],
        [    39.5525,   -188.9732,   2500.5993,    251.2284],
        [    39.5525,   -188.9732,  72001.    ,    483.0622]],

       [[  -239.5126,    -92.2893, -71998.8026,      0.    ],
        [  -239.5126,    -92.2893,  -2500.5993,    231.8339],
        [  -239.5126,    -92.2893,   1500.7991,    245.1704],
        [  -239.5126,    -92.2893,   2500.5993,    251.2284],
        [  -239.5126,    -92.2893,  72001.    ,    483.0622]],

::

     
     82 /**
     83 shortnorm
     84 ------------
     85 
     86 range of short is -32768 to 32767
     87 Expect no positions out of range, as constrained by the geometry are bouncing on,
     88 but getting times beyond the range eg 0.:100 ns is expected
     89 
     90 **/
     91 
     92 __device__ short shortnorm( float v, float center, float extent )
     93 {
     94     int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
     95     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
     96 }   



After re-simulating to fix the time domain and using automated rule of thumb to 
set the timedomain based on geometry extent the -ve times are gone and 
the visualized propagation looks more reasonable::

    TMAX=-1 ts 10 
    TMAX=-1 tv 10 






