invert_trs_nmat4triple_warnings_deferredCreateGPartsTest
===========================================================

See Also
-----------

* ~/jnu/issues/geocache-j2102-shakedown.rst


Overview
----------

1. conversion of latest geometry with geocache-j21 gave loadsa warning output from nmat4triple invert_trs
2. collected the placement transforms that are causing this, switched on with envvar OPTICKS_GGEO_SAVE_MISMATCH_PLACEMENTS

   * /tmp/blyth/opticks/GGeo__deferredCreateGParts/mm0/mismatch_placements.npy

3. identify which geometry is causing this, using ~/opticks/ggeo/tests/mismatch_placements.py 
   accessing ndIdx identity info planted into the spare column 4 of the transforms.

   * all from global geometry, that is not optically relevant 
  
4. get the names of the 42 lv/pv using cat.py in the geocache dir::

   export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x344f8d0_PV.732a5daf83a7153b316a2013fcfb1fc2
   kcd 
   cd GNodeLib

   epsilon:GNodeLib blyth$ cat.py -s 66464,66471,66479,66486,66560,66575,66591,66606,66771,66786,66796,66799,66800,66811,66814,66815,66852,66863,66867,66878,67042,67057,67572,67587,67604,67612,67619,67627,67641,67656,68003,68007,68018,68022,68059,68071,68074,68086,68090,68091,68105,68106 all_volume_PVNames.txt
    66464 66465 GLb2.up07_HBeam_phys0x34a9c20
    66471 66472 GLb2.up07_HBeam_phys0x34aa390
    66479 66480 GLb2.up07_HBeam_phys0x34aac10
    ...
    66867 66868 GLb1.bt06_HBeam_phys0x34d3190
    66878 66879 GLb1.bt06_HBeam_phys0x34d3d40
    67042 67043 GZ1.A01_02_HBeam_phys0x34e1c40
    67057 67058 GZ1.A01_02_HBeam_phys0x34e2c30
    67572 67573 ZC2.A03_B04_HBeam_phys0x3515550
    ...
    67627 67628 ZC2.A04_B05_HBeam_phys0x3519480
    67641 67642 ZC2.A05_B06_HBeam_phys0x351aab0
    67656 67657 ZC2.A05_B06_HBeam_phys0x351b9b0
    68003 68004 lSteel_phys0x353ac50
    68007 68008 lSteel_phys0x353b010
    68018 68019 lSteel_phys0x353ba60
    ...
     68091 68092 lSteel_phys0x35402f0
    68105 68106 lSteel_phys0x3541010
    68106 68107 lSteel_phys0x3541100
    epsilon:GNodeLib blyth$ 


5. added ggeo/tests/deferredCreateGPartsTest.cc to read those transforms and and create nmat4triple from them, 
   this succeeds to reproduce the warning output in a quick test

   * reveals the warnings were caused by "crazy" diffFractional of 2 which happens when
     have two small values that straddle the epsilon value : causing one to be pinned to zero
     and the other not.  That results in the average being a/2 or b/2 yielding
     a values of 2 for the fractional difference.


6. fixed nglmext::compDiff2 with epsilon_horizon to avoid the crazies from epsilon straddlers


fixed nglmext::compDiff2
----------------------------

::

     746 /**
     747 nglmext::compDiff2
     748 ---------------------
     749 
     750 fractional:false
     751     returns absolute difference between a and b where a or b values 
     752     within epsilon of zero are set to zero 
     753 
     754 fractional:true
     755     returns the fractional:false value divided by the average of a and b    
     756 
     757 **/
     758 
     759 float nglmext::compDiff2(const float a_ , const float b_, bool fractional, float u_epsilon )
     760 {
     761     float a = fabsf(a_) < u_epsilon  ? 0.f : a_ ;
     762     float b = fabsf(b_) < u_epsilon  ? 0.f : b_ ;
     763     float d = 0.f ;
     764 
     765     if( fractional )
     766     {
     767         float avg = (a+b)/2.f ;
     768         bool epsilon_horizon = fabsf(a_) < u_epsilon || fabsf(b_) < u_epsilon  ;
     769         d = epsilon_horizon || avg == 0.f ? 0.f : fabsf(a - b)/avg ;
     770     }
     771     else
     772     {
     773         d = fabsf(a - b);
     774     }
     775 
     776     return d ;
     777 }
     778 






nmat4triple invert_trs warnings
------------------------------------

::

    2021-02-19 14:34:12.184 INFO  [14436267] [*GParts::Create@214] [  deferred creation from GPts
    2021-02-19 14:34:12.184 INFO  [14436267] [*GParts::Create@220]  num_pt 3084
    2021-02-19 14:34:12.253 ERROR [14436267] [nglmext::invert_trs@488] ngmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  epsilon 1e-05 diff 0.00195312 diff2 0.00195312 diffFractional 2 diffFractionalMax 0.001
             A  0.565  -0.743   0.359   0.000 
                0.627   0.669   0.398   0.000 
               -0.536   0.000   0.844   0.000 
                0.002  -0.000 -20250.002   1.000 
    np.fromstring("0.564966 -0.743145 0.358538 0 0.627458 0.669131 0.398197 0 -0.535827 1.38778e-16 0.844328 0 0.00195312 -2.37277e-12 -20250 1 ", dtype=np.float32, sep=" ").reshape(4,4) 

            B   0.565  -0.743   0.359  -0.000 
                0.627   0.669   0.398   0.000 
               -0.536  -0.000   0.844  -0.000 
                0.000  -0.000 -20250.000   1.000 
    np.fromstring("0.564966 -0.743145 0.358538 -0 0.627458 0.669131 0.398197 0 -0.535827 -0 0.844328 -0 0.000326724 -0.000437193 -20250 1 ", dtype=np.float32, sep=" ").reshape(4,4) 

    [  0.564966:  0.564966:         0:         0][ -0.743145: -0.743145:5.96046e-08:-8.0206e-08][  0.358538:  0.358538:         0:         0][         0:        -0:         0:         0]
    [  0.627458:  0.627458:         0:         0][  0.669131:  0.669131:5.96046e-08:8.90778e-08][  0.398197:  0.398197:2.98023e-08:7.48432e-08][         0:         0:         0:         0]
    [ -0.535827: -0.535827:5.96046e-08:-1.11239e-07][1.38778e-16:        -0:         0:         0][  0.844328:  0.844328:5.96046e-08:7.05942e-08][         0:        -0:         0:         0]
    [**0.00195312:0.000326724: 0.0016264:   1.42676**][-2.37277e-12:-0.000437193:0.000437193:        -2][    -20250:    -20250:0.00195312:-9.64506e-08][         1:         1:         0:         0]

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGINT
      * frame #0: 0x00007fff61fa7b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff62172080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff61eb56fe libsystem_c.dylib`raise + 26
        frame #3: 0x000000010a44020f libNPY.dylib`nmat4triple::nmat4triple(this=0x00007ffeefbfba40, t_=0x000000011fe10e08)0> const&) at NGLMExt.cpp:905
        frame #4: 0x000000010a44024d libNPY.dylib`nmat4triple::nmat4triple(this=0x00007ffeefbfba40, t_=0x000000011fe10e08)0> const&) at NGLMExt.cpp:901
        frame #5: 0x000000010a442800 libNPY.dylib`nmat4triple::make_transformed(src=0x000000025b55d350, txf=0x000000011fe10e08, reverse=true, (null)="GParts::applyPlacementTransform")0> const&, bool, char const*) at NGLMExt.cpp:1115
        frame #6: 0x00000001099e8c00 libGGeo.dylib`GParts::applyPlacementTransform(this=0x000000025b55d240, placement=0x000000011fe10e08, verbosity=0)0> const&, unsigned int) at GParts.cc:1055
        frame #7: 0x00000001099e7450 libGGeo.dylib`GParts::Create(pts=0x00000001b3d84cf0, solids=size=137) at GParts.cc:240
        frame #8: 0x0000000109a66c33 libGGeo.dylib`GGeo::deferredCreateGParts(this=0x0000000115456fa0) at GGeo.cc:1328
        frame #9: 0x0000000109a67228 libGGeo.dylib`GGeo::postDirectTranslation(this=0x0000000115456fa0) at GGeo.cc:581
        frame #10: 0x000000010001574b OKX4Test`main(argc=13, argv=0x00007ffeefbfe4d0) at OKX4Test.cc:113
        frame #11: 0x00007fff61e57015 libdyld.dylib`start + 1
    (lldb) 


