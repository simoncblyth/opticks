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



