hit-flags-zero-pmtid-with-xanalytic-and-iidentity
===================================================

issue
---------

With analytic geometry (using --xanalytic) note that get pmtid 0 for all hits,  whilst 
triangulated geometry (without --xanalytic) gives expected pmtid. 

Analytic geometry was using aiidentity, have excluded that via WITH_AII, 
and adopted iidentity.

* aii : (num_instances, 4)
* ii : ( num_instance, num_volumes, 4 )


::

     i      32 post ( -12995.476  8907.965 11091.666     101.583) flgs     -25       0 67305985    2130
     i      33 post (  11271.271-15517.808 -2011.677     123.457) flgs     -25       0 67305985    2146
     i      34 post (  -8778.634-17186.625   855.321     101.076) flgs     -25       0 67305985    2162
     i      35 post ( -19187.506 -2456.433   981.842     118.980) flgs     -29       0 67305985    2130
     i      36 post (  13017.426 -9005.810-10995.156     105.313) flgs     -25       0 67305985    2146
     i      37 post ( -18017.471  5613.289 -3842.849     100.840) flgs     -25       0 67305985    2114


The signed bnd index -25, -29 are 1-based : so 24 28 on the 0-based blib list::


    blyth@localhost 1]$ ~/anaconda2/bin/python ~/opticks/ana/blib.py $GC -s 24,26,28
     nbnd  35 nmat  39 nsur  34 
     24 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
     26 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
     28 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 

* are missing bnd -27 : so no Hamamatsu hits ?

::

    epsilon:1 blyth$ ~/opticks/ana/blib.py $GC | grep photocathode
     24 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
     26 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
     28 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     33 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 



Dumping from the bounds program shows the identity info is there GPU side, overwrite somewhere ?
---------------------------------------------------------------------------------------------------

::

    2020-07-21 02:29:13.972 INFO  [388191] [OContext::launch@779] COMPILE time: 7.00001e-06
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 0 instance_index 0 primitive_count 0 primIdx 1 identity (       1      12       1       0 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 1 instance_index 0 primitive_count 5 primIdx 1 identity (  173923      38      22  300000 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 2 instance_index 0 primitive_count 6 primIdx 1 identity (   68251      24      15       0 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 3 instance_index 0 primitive_count 6 primIdx 1 identity (   68257      30      15       1 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 4 instance_index 0 primitive_count 6 primIdx 1 identity (  301927      47      15   30000 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 9 instance_index 0 primitive_count 130 primIdx 1 identity (      11       6       8       0 ) 

    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 0 instance_index_test 10 primitive_count 0 primIdx 1 identity (       1      12       1       0 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 1 instance_index_test 10 primitive_count 5 primIdx 1 identity (  173973      38      22  300010 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 2 instance_index_test 10 primitive_count 6 primIdx 1 identity (   68335      24      15      14 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 3 instance_index_test 10 primitive_count 6 primIdx 1 identity (   68467      30      15      36 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 4 instance_index_test 10 primitive_count 6 primIdx 1 identity (  301987      47      15   30010 ) 
    // intersect_analysic.cu:bounds WITH_PRINT_IDENTITY_BOUNDS repeat_index 9 instance_index_test 10 primitive_count 130 primIdx 1 identity (    1314       6       8       2 ) 

    2020-07-21 02:29:18.311 INFO  [388191] [OContext::launch@786] PRELAUNCH time: 4.33873
    2020-07-21 02:29:18.311 INFO  [388191] [OPropagator::prelaunch@195] 0 : (0;0,0) 



need quicker debug turnaround that tds  : added okt using the geocache and gensteps created by tds
-----------------------------------------------------------------------------------------------------

* this avoids geometry conversion + Geant4 initialization(voxeling) time 

::

    [blyth@localhost 1]$ t okt
    okt is a function
    okt () 
    { 
        type $FUNCNAME;
        [ -z "$OPTICKS_KEY" ] && echo $msg MISSING MANDATORY OPTICKS_KEY envvar && return 1;
        [ "$(which OKTest 2>/dev/null)" == "" ] && echo $msg missing opticks env use : oe- && return 2;
        elog;
        local args="OKTest --xanalytic --save --dbggsload --dumphit --dbggsdir /tmp/$USER/opticks/dbggs --printenabled --pindex ${P:-1000} ";
        if [ -z "$BP" ]; then
            H="";
            B="";
            T="-ex r";
        else
            H="-ex \"set breakpoint pending on\"";
            B="";
            for bp in $BP;
            do
                B="$B -ex \"break $bp\" ";
            done;
            T="-ex \"info break\" -ex r";
        fi;
        local iwd=$PWD;
        local dir=/tmp/$USER/opticks/okt;
        mkdir -p $dir;
        cd_func $dir;
        local runline="gdb $H $B $T --args $args ";
        echo $runline;
        date;
        eval $runline;
        date
    }




Switching on --printenabled with --pindex 1000 reveals CRAZY numParts 511 blowout
----------------------------------------------------------------------------------

::


    [blyth@localhost okt]$ P=1000 okt
    ...

    2020-07-21 18:56:32.363 NONE  [86278] [OPropagator::launch@250]  _prelaunch 1 m_width 11235 m_height 1
    2020-07-21 18:56:32.363 INFO  [86278] [OPropagator::launch@267] LAUNCH NOW -
    // evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 21 numParts 511 perfect tree height 8 exceeds current limit
    // evaluative_csg repeat_index 3 tranOffset 30 numParts 511 perfect tree height 8 exceeds current limit
    2020-07-21 18:56:32.646 INFO  [86278] [OPropagator::launch@276] LAUNCH DONE
    2020-07-21 18:56:32.647 INFO  [86278] [OPropagator::launch@278] 0 : (0;11235,1) 

::

    [blyth@localhost okt]$ P=4 okt    ## bounds dumping for primIdx 4 shows the same crazy numParts 
    ...

    2020-07-21 06:20:39.265 INFO  [284797] [OContext::launch@783]  entry 0 width 0 height 0  --printenabled  printLaunchIndex ( 4 0 0)
    2020-07-21 06:20:39.311 INFO  [284797] [OContext::launch@796] VALIDATE time: 0.045615
    2020-07-21 06:20:39.311 INFO  [284797] [OContext::launch@803] COMPILE time: 7e-06
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS repeat_index 0 primIdx 4 primFlag 101 partOffset 4 tranOffset 4 numParts 1 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 0 typecode 12 boundary 4 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS repeat_index 1 primIdx 4 primFlag 101 partOffset 6 tranOffset 4 numParts 1 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 0 typecode 12 boundary 19 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS repeat_index 2 primIdx 4 primFlag 101 partOffset 22 tranOffset 11 numParts 15 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 0 typecode 2 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 1 typecode 1 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 2 typecode 12 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 3 typecode 1 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 4 typecode 12 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 5 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 6 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 7 typecode 5 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 8 typecode 15 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 9 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 10 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 11 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 12 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 13 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 14 typecode 0 boundary 24 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS repeat_index 3 primIdx 4 primFlag 101 partOffset 38 tranOffset 21 numParts 511    ##### CRAZY numParts
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 0 typecode 2 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 1 typecode 1 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 2 typecode 12 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 3 typecode 1 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 4 typecode 12 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 5 typecode 0 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 6 typecode 0 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 7 typecode 1 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 8 typecode 12 boundary 26 
    // intersect_analysic.cu:bounds WITH_PRINT_PARTS p 9 typecode 0 boundary 26 



Check the primBuf
------------------------


::

     24 struct Prim
     25 {
     26     __device__ int partOffset() const { return  q0.i.x ; }
     27     __device__ int numParts()   const { return  q0.i.y < 0 ? -q0.i.y : q0.i.y ; }
     28     __device__ int tranOffset() const { return  q0.i.z ; }
     29     __device__ int planOffset() const { return  q0.i.w ; }
     30     __device__ int primFlag()   const { return  q0.i.y < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE ; }
     31 
     32     quad q0 ;
     33 
     34 };
     35 



::

    epsilon:1 blyth$ inp GParts/?/primBuffer.npy -l
    a :                                      GParts/0/primBuffer.npy :             (374, 4) : 8758dbcb7bc9fb41572f227925582798 : 20200719-2129 
    b :                                      GParts/1/primBuffer.npy :               (5, 4) : 2280f02492cadf1cac6eb6e3080058c9 : 20200719-2129 
    c :                                      GParts/2/primBuffer.npy :               (6, 4) : 3a439307b3494d399d6b889bb2d5fcc0 : 20200719-2129 
    d :                                      GParts/3/primBuffer.npy :               (6, 4) : e0bc7b4fdd932199e0b7815a4a6da62c : 20200719-2129 
    e :                                      GParts/4/primBuffer.npy :               (6, 4) : b3d86640eae7f3da0db1e6878921aca4 : 20200719-2129 
    f :                                      GParts/5/primBuffer.npy :               (1, 4) : 4d871a51138cc646de8d2831e2ec299b : 20200719-2129 
    g :                                      GParts/6/primBuffer.npy :               (1, 4) : 4f07a8b7535e2d3c7238b970cc45d2d7 : 20200719-2129 
    h :                                      GParts/7/primBuffer.npy :               (1, 4) : 4d871a51138cc646de8d2831e2ec299b : 20200719-2129 
    i :                                      GParts/8/primBuffer.npy :               (1, 4) : 4f07a8b7535e2d3c7238b970cc45d2d7 : 20200719-2129 
    j :                                      GParts/9/primBuffer.npy :             (130, 4) : 73df7a651dd474a5d533e614d64b91fe : 20200719-2129 

    In [1]: a
    Out[1]: 
    array([[   0,    1,    0,    0],
           [   1,    1,    1,    0],
           [   2,    1,    2,    0],
           ...,
           [1909,    3,  964,    0],
           [1912,    1,  965,    0],
           [1913,    3,  966,    0]], dtype=int32)

    In [2]: a[:,1]   ## note these are all complete binary tree sizes  1,3,7,15
    Out[2]: 
    array([ 1,  1,  1,  1,  1,  3,  3,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  3,  3,  3,  3, 15, 15,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 15, 15, 15, 15, 15, 15, 15, 15,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  1,  3,  1,  3], dtype=int32)

    In [3]: b[:,1]
    Out[3]: array([3, 1, 1, 1, 1], dtype=int32)

    In [4]: c[:,1]
    Out[4]: array([ 1,  7,  7,  7, 15, 15], dtype=int32)

    In [5]: d[:,1]
    Out[5]: array([  1,   7,  15,  15, 511, 511], dtype=int32)      ## OUCH 511 for the last 2 volumes, I would have expected 15  (as less than 8 leaves)

    In [12]: d
    Out[12]: 
    array([[  0,   1,   0,   0],
           [  1,   7,   1,   0],
           [  8,  15,   5,   0],
           [ 23,  15,  13,   0],
           [ 38, 511,  21,   0],
           [549, 511,  30,   0]], dtype=int32)



    In [6]: e[:,1]
    Out[6]: array([1, 7, 3, 3, 7, 7], dtype=int32)

    In [7]: f[:,1]
    Out[7]: array([3], dtype=int32)

    In [8]: g[:,1]
    Out[8]: array([31], dtype=int32)

    In [9]: h[:,1]
    Out[9]: array([3], dtype=int32)

    In [10]: i[:,1]
    Out[10]: array([31], dtype=int32)

    In [11]: j[:,1]
    Out[11]: 
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1], dtype=int32)

    In [12]: 




Check the GPts buffers
------------------------

* these inputs to the GParts all look as expected


::

    epsilon:GPts blyth$ inp ?/iptBuffer.npy -l
    a :                                              0/iptBuffer.npy :             (374, 4) : cfcb7b3c1f2314b02ed20609f687c52f : 20200719-2129 
    b :                                              1/iptBuffer.npy :               (5, 4) : 0a7c1e906a6a3913f3bcfe3ab4d40dd7 : 20200719-2129 
    c :                                              2/iptBuffer.npy :               (6, 4) : 42761fa2b500a8fd70d9f67416f9c916 : 20200719-2129 
    d :                                              3/iptBuffer.npy :               (6, 4) : a7d635662dee3dc1ea006fd36a18763f : 20200719-2129 
    e :                                              4/iptBuffer.npy :               (6, 4) : d0650e08593ea37ed79aab92cab13604 : 20200719-2129 
    f :                                              5/iptBuffer.npy :               (1, 4) : 547da34217547f78916d7ec9f136ed9a : 20200719-2129 
    g :                                              6/iptBuffer.npy :               (1, 4) : d26bda9e14e82bf4a256d1098084e692 : 20200719-2129 
    h :                                              7/iptBuffer.npy :               (1, 4) : 07fdae2d906fed39fedc7e95ca7136d5 : 20200719-2129 
    i :                                              8/iptBuffer.npy :               (1, 4) : 2ff7c7568240328b81716a99ab93f5ef : 20200719-2129 
    j :                                              9/iptBuffer.npy :             (130, 4) : 8c925a62dc2af568e967e927da9b52b5 : 20200719-2129 

    In [1]: d   (6,4) (num_volumes, num_qty)
    Out[1]: 
    array([[   35, 68256,    35,     0],
           [   30, 68257,    30,     1],
           [   34, 68258,    34,     2],
           [   33, 68259,    33,     3],
           [   31, 68260,    31,     4],
           [   32, 68261,    32,     5]], dtype=int32)

          ## lvIdx ndIdx  csgIdx             csgIdx 31 and 32 are the ones with the problem 

    epsilon:GPts blyth$ cat 3/GPts.txt
    Water///Water
    Water///Acrylic
    Water///Pyrex
    Pyrex///Pyrex
    Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum
    Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum


    epsilon:GPts blyth$ inp ?/plcBuffer.npy -l   ## only 0 and 9 have non-identity transforms as expected

    a :                                              0/plcBuffer.npy :          (374, 4, 4) : b915529d5802b34c5805e865d9ba8535 : 20200719-2129 
    b :                                              1/plcBuffer.npy :            (5, 4, 4) : 22634081e8f79c10be247f677e23ef59 : 20200719-2129 
    c :                                              2/plcBuffer.npy :            (6, 4, 4) : 07913323c98448be590fd704a2d23c5b : 20200719-2129 
    d :                                              3/plcBuffer.npy :            (6, 4, 4) : 07913323c98448be590fd704a2d23c5b : 20200719-2129 
    e :                                              4/plcBuffer.npy :            (6, 4, 4) : 07913323c98448be590fd704a2d23c5b : 20200719-2129 
    f :                                              5/plcBuffer.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200719-2129 
    g :                                              6/plcBuffer.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200719-2129 
    h :                                              7/plcBuffer.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200719-2129 
    i :                                              8/plcBuffer.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200719-2129 
    j :                                              9/plcBuffer.npy :          (130, 4, 4) : a1df8dac85426d19574f86b47e4da0b3 : 20200719-2129 




Check the GParts : ridx 3 affliction shows
---------------------------------------------

::

    epsilon:GParts blyth$ wc -l ?/GParts.txt
        1916 0/GParts.txt
           7 1/GParts.txt
          52 2/GParts.txt
        1060 3/GParts.txt
          28 4/GParts.txt
           3 5/GParts.txt
          31 6/GParts.txt
           3 7/GParts.txt
          31 8/GParts.txt
         130 9/GParts.txt
        3261 total


    epsilon:GParts blyth$ inp 2/*.npy 
    a :                                             2/partBuffer.npy :           (52, 4, 4) : f90b2e01df8a10734f46fd1ef724c368 : 20200719-2129 
    b :                                             2/tranBuffer.npy :        (19, 3, 4, 4) : 71252d65f8f924e12439823817a64084 : 20200719-2129 
    c :                                              2/idxBuffer.npy :               (6, 4) : 214787248c9a4dc22bdb294dcf89a1a8 : 20200719-2129 
    d :                                             2/primBuffer.npy :               (6, 4) : 3a439307b3494d399d6b889bb2d5fcc0 : 20200719-2129 


    epsilon:GParts blyth$ inp 3/*.npy 
    a :                                             3/partBuffer.npy :         (1060, 4, 4) : c37a5b4289730a8bdfc220916dd5961a : 20200719-2129 
    b :                                             3/tranBuffer.npy :        (39, 3, 4, 4) : 454db14f2d06356fe113256b0eb8f47a : 20200719-2129 
    c :                                              3/idxBuffer.npy :               (6, 4) : b9b169020a59d58357c30e84259d7dcb : 20200719-2129 
    d :                                             3/primBuffer.npy :               (6, 4) : e0bc7b4fdd932199e0b7815a4a6da62c : 20200719-2129 


    ## sizes of part buffers should be from sums of the complete binary tree size of the csg solids

    epsilon:GParts blyth$ inp ?/partBuffer.npy -l
    a :                                             0/partBuffer.npy :         (1916, 4, 4) : 19744c8231410146db3dc410dd7ce3a2 : 20200719-2129 
    b :                                             1/partBuffer.npy :            (7, 4, 4) : 6e15df9fb2b68f59bdbff2fbf440d330 : 20200719-2129 
    c :                                             2/partBuffer.npy :           (52, 4, 4) : f90b2e01df8a10734f46fd1ef724c368 : 20200719-2129 
    d :                                             3/partBuffer.npy :         (1060, 4, 4) : c37a5b4289730a8bdfc220916dd5961a : 20200719-2129 
    e :                                             4/partBuffer.npy :           (28, 4, 4) : 0ccc935013622f72b1cb65c4f0e9079e : 20200719-2129 
    f :                                             5/partBuffer.npy :            (3, 4, 4) : 3ff18ec9f9cdccece2a0d285b1c76641 : 20200719-2129 
    g :                                             6/partBuffer.npy :           (31, 4, 4) : 8b34d3820e39902e43d5a734b1d6caec : 20200719-2129 
    h :                                             7/partBuffer.npy :            (3, 4, 4) : bb9615e1f4163de171217aa5c72514d5 : 20200719-2129 
    i :                                             8/partBuffer.npy :           (31, 4, 4) : 1aec5b80ba1258f652a779716a846b1d : 20200719-2129 
    j :                                             9/partBuffer.npy :          (130, 4, 4) : 4dcc2d5c7b026905917868dd5243ae02 : 20200719-2129 




::


     216 GParts* GParts::Create(const GPts* pts, const std::vector<const NCSG*>& solids, unsigned verbosity) // static
     217 {
     218     LOG(LEVEL) << "[  deferred creation from GPts" ;
     219 
     220     GParts* com = new GParts() ;
     221 
     222     unsigned num_pt = pts->getNumPt();
     223 
     224     LOG(LEVEL) << " num_pt " << num_pt ;
     225 
     226     for(unsigned i=0 ; i < num_pt ; i++)
     227     {
     228         const GPt* pt = pts->getPt(i);
     229         int   lvIdx = pt->lvIdx ;
     230         int   ndIdx = pt->ndIdx ;
     231         const std::string& spec = pt->spec ;
     232         const glm::mat4& placement = pt->placement ;
     233         assert( lvIdx > -1 );
     234 
     235         const NCSG* csg = unsigned(lvIdx) < solids.size() ? solids[lvIdx] : NULL ;
     236         assert( csg );
     237 
     238         //  X4PhysicalVolume::convertNode
     239         GParts* parts = GParts::Make( csg, spec.c_str(), ndIdx );
     240         //parts->setVolumeIndex(ndIdx); 
     241 
     242         // GMergedMesh::mergeVolume
     243         // GMergedMesh::mergeVolumeAnalytic
     244         parts->applyPlacementTransform( placement );
     245 
     246         com->add( parts, verbosity );
     247     }
     248     LOG(LEVEL) << "]" ;
     249     return com ;
     250 }


GParts transitions NCSG tree of the solid to node level and provides concatenated persistency::

     473 GParts* GParts::Make( const NCSG* tree, const char* spec, unsigned ndIdx )
     474 {
     475     assert(spec);
     476 
     477     bool usedglobally = tree->isUsedGlobally() ;   // see opticks/notes/issues/subtree_instances_missing_transform.rst
     478     assert( usedglobally == true );  // always true now ?   
     479 
     480     NPY<unsigned>* tree_idxbuf = tree->getIdxBuffer() ;   // (1,4) identity indices (index,soIdx,lvIdx,height)
     481     NPY<float>*   tree_tranbuf = tree->getGTransformBuffer() ;
     482     NPY<float>*   tree_planbuf = tree->getPlaneBuffer() ;
     483     assert( tree_tranbuf );
     484 
     485     NPY<unsigned>* idxbuf = tree_idxbuf->clone()  ;   // <-- lacking this clone was cause of the mystifying repeated indices see notes/issues/GPtsTest             
     486     NPY<float>* nodebuf = tree->getNodeBuffer();       // serialized binary tree
     487     NPY<float>* tranbuf = usedglobally                 ? tree_tranbuf->clone() : tree_tranbuf ;
     488     NPY<float>* planbuf = usedglobally && tree_planbuf ? tree_planbuf->clone() : tree_planbuf ;
     489 
     490 
     491     // overwrite the cloned idxbuf swapping the tree index for the ndIdx 
     492     // as being promoted to node level 
     493     {
     494         assert( idxbuf->getNumItems() == 1 );
     495         unsigned i=0u ;
     496         unsigned j=0u ;
     497         unsigned k=0u ;
     498         unsigned l=0u ;
     499         idxbuf->setUInt(i,j,k,l, ndIdx);
     500     }
     ...
     563     GItemList* lspec = GItemList::Repeat("GParts", spec, ni, reldir) ;
     564 
     565     GParts* pts = new GParts(idxbuf, nodebuf, tranbuf, planbuf, lspec) ;
     566 
     567     //pts->setTypeCode(0u, root->type);   //no need, slot 0 is the root node where the type came from
     568 
     569     pts->setCSG(tree);
     570 
     571     return pts ;
     572 }


::

    epsilon:1 blyth$ cat GMeshLib/MeshUsage.txt | grep Hamamatsu
        30 ( v  484 f  960 ) :    5000 :    2420000 :    4800000 : HamamatsuR12860sMask0x3291550
        31 ( v  302 f  600 ) :    5000 :    1510000 :    3000000 : HamamatsuR12860_PMT_20inch_inner1_solid0x32a8f30
        32 ( v  688 f 1364 ) :    5000 :    3440000 :    6820000 : HamamatsuR12860_PMT_20inch_inner2_solid0x32a91b0
        33 ( v  820 f 1624 ) :    5000 :    4100000 :    8120000 : HamamatsuR12860_PMT_20inch_body_solid_1_90x32b7d70
        34 ( v  820 f 1624 ) :    5000 :    4100000 :    8120000 : HamamatsuR12860_PMT_20inch_pmt_solid_1_90x329ed30
        35 ( v   50 f   96 ) :    5000 :     250000 :     480000 : HamamatsuR12860sMask_virtual0x3290560


    epsilon:GMeshLibNCSG blyth$ cat 32/meta.json 
    {"balanced":0,"lvname":"HamamatsuR12860_PMT_20inch_inner2_log0x32a9750","soname":"HamamatsuR12860_PMT_20inch_inner2_solid0x32a91b0"}epsilon:GMeshLibNCSG blyth$ 
    epsilon:GMeshLibNCSG blyth$ cat 31/meta.json 
    {"balanced":0,"lvname":"HamamatsuR12860_PMT_20inch_inner1_log0x32a9620","soname":"HamamatsuR12860_PMT_20inch_inner1_solid0x32a8f30"}epsilon:GMeshLibNCSG blyth$ 
    epsilon:GMeshLibNCSG blyth$ 


    epsilon:GMeshLibNCSG blyth$ inp 31/*.npy 32/*.npy 
    a :                                              31/srcnodes.npy :          (511, 4, 4) : b5f789f1963239d38971d5b21d3c9838 : 20200719-2129 
    b :                                              32/srcnodes.npy :          (511, 4, 4) : 3eb56acf2d98394435d6a1addafdd637 : 20200719-2129 
    c :                                         31/srctransforms.npy :           (17, 4, 4) : fc8c1e11dc40a0ffb91a469e43ac30e4 : 20200719-2129 
    d :                                         32/srctransforms.npy :           (17, 4, 4) : fc8c1e11dc40a0ffb91a469e43ac30e4 : 20200719-2129 
    e :                                                31/srcidx.npy :               (1, 4) : fda466224f5f008aa02332c0e249e892 : 20200719-2129 
    f :                                                32/srcidx.npy :               (1, 4) : 8f7c79a693352ec76561d895bc9e78da : 20200719-2129 




x4gen with x031 reproduces the NCSG with 511 nodes
----------------------------------------------------


Get x4gen-- to work and build the generated x031.cc x032.cc::

    epsilon:tests blyth$ x031
    2020-07-21 13:30:40.573 INFO  [10693791] [Opticks::init@405] INTEROP_MODE hostname epsilon.local
    2020-07-21 13:30:40.573 INFO  [10693791] [Opticks::init@414]  non-legacy mode : ie mandatory keyed access to geometry, opticksaux 
    2020-07-21 13:30:40.576 INFO  [10693791] [BOpticksResource::setupViaKey@828] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x30d4f90_PV.ad026c799f5511ddb91eb379efa84bc4
                     exename  : OKX4Test
             current_exename  : x031
                       class  : X4PhysicalVolume
                     volname  : lWorld0x30d4f90_PV
                      digest  : ad026c799f5511ddb91eb379efa84bc4
                      idname  : OKX4Test_lWorld0x30d4f90_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-07-21 13:30:40.577 INFO  [10693791] [Opticks::loadOriginCacheMeta@1838]  cachemetapath /usr/local/opticks/geocache/OKX4Test_lWorld0x30d4f90_PV_g4live/g4ok_gltf/ad026c799f5511ddb91eb379efa84bc4/1/cachemeta.json
    2020-07-21 13:30:40.577 INFO  [10693791] [NMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "argline": "/usr/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /usr/local/opticks/tds_ngt_pcnk.gdml -D ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20200719_212814",
        "runfolder": "OKX4Test",
        "runlabel": "R0_cvd_",
        "runstamp": 1595190494
    }
    2020-07-21 13:30:40.577 INFO  [10693791] [Opticks::loadOriginCacheMeta@1842]  gdmlpath /usr/local/opticks/tds_ngt_pcnk.gdml
    2020-07-21 13:30:40.593 INFO  [10693791] [*NTreeBalance<nnode>::create_balanced@59] op_mask union intersection 
    2020-07-21 13:30:40.593 INFO  [10693791] [*NTreeBalance<nnode>::create_balanced@60] hop_mask union intersection 
    2020-07-21 13:30:40.593 FATAL [10693791] [*NTreeBalance<nnode>::create_balanced@101] balancing trees of this structure not implemented
    2020-07-21 13:30:40.593 ERROR [10693791] [NNodeNudger::init@88] NNodeNudger::brief root.treeidx   0 num_prim  9 num_coincidence  7 num_nudge  2 
    NCSGList::savesrc csgpath /tmp/blyth/opticks/x4gen/x031 verbosity 0 numTrees 2
    2020-07-21 13:30:40.603 INFO  [10693791] [NCSG::savesrc@305]  treedir_ /tmp/blyth/opticks/x4gen/x031/0
    2020-07-21 13:30:40.604 INFO  [10693791] [NCSG::savesrc@305]  treedir_ /tmp/blyth/opticks/x4gen/x031/1
    analytic=1_csgpath=/tmp/blyth/opticks/x4gen/x031
    2020-07-21 13:30:40.605 INFO  [10693791] [X4CSG::dumpTestMain@272] X4CSG::dumpTestMain


    epsilon:1 blyth$ inp *.npy
    a :                                                 srcnodes.npy :          (511, 4, 4) : b5f789f1963239d38971d5b21d3c9838 : 20200721-1330 
    b :                                            srctransforms.npy :           (17, 4, 4) : fc8c1e11dc40a0ffb91a469e43ac30e4 : 20200721-1330 
    c :                                                   srcidx.npy :               (1, 4) : 520e4c77ebd5ac14a93eed531c3d9cd2 : 20200721-1330 

    In [1]: pwd
    Out[1]: u'/private/tmp/blyth/opticks/x4gen/x031/1'





