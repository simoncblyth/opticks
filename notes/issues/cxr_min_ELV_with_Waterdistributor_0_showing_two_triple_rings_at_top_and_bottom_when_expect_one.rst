cxr_min_ELV_with_Waterdistributor_0_showing_two_triple_rings_at_top_and_bottom_when_expect_one
==================================================================================================

Overview
----------

Found the cause to be that the four Waterdistributor in the global CSGSolid
have duplicated CSGPrim names.

This probably is just an inconvenience that is best fixed
by using more distinct G4VSolid names in the source geometry ?

This is because its too awkward to add ordinal selection to
ELV, because are potentially selecting loads of names all at once.


Reproduce
------------


ELV::

     15 #elv_name=cosmetic
     16 elv_name=wd_debug
     17 #elv_name=wd_lower
     18 #elv_name=wd_upper
     19 
     20 if [ "$GEOM" == "J25_4_0_opticks_Debug" -a "$elv_name" == "wd_debug" ]; then
     21 
     22    elv=""
     23    elv="$elv,Waterdistributor_0"      ## HUH : BOTH LOWER AND UPPER IN ONE VOL ?
     24    #elv="$elv,Waterdistributor_1"      ## HUH : BOTH LOWER AND UPPER IN ONE VOL ?
     25    #elv="$elv,Waterdistributor_2"      ## HUH : BOTH LOWER AND UPPER IN ONE VOL ?
     26    #elv="$elv,Waterdistributor_3"      ## HUH : BOTH LOWER AND UPPER IN ONE VOL ?
     27 



One volume is ELV selected, CSGFoundry=INFO cxr_min.sh
------------------------------------------------------------

::

    (ok) A[blyth@localhost junosw]$ CSGFoundry=INFO cxr_min.sh


    2025-09-30 09:48:42.598 INFO  [2914887] [CSGFoundry::CopySelect@3089] [
    2025-09-30 09:48:42.598 INFO  [2914887] [CSGFoundry::CopySelect@3091]    -        37 340 : 0000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

    2025-09-30 09:48:42.598 INFO  [2914887] [CSGFoundry::CopySelect@3092] CSGFoundry::descELV elv.num_bits 340 num_include 1 num_exclude 339 is_all_set 0
    INCLUDE:1

    p: 37:midx: 37:mn:Waterdistributor_0
    EXCLUDE:339


::

    3087 CSGFoundry* CSGFoundry::CopySelect(const CSGFoundry* src, const SBitSet* elv )
    3088 {
    3089     LOG(LEVEL) << "[" ;
    3090     assert(elv);
    3091     LOG(LEVEL) << elv->desc() << std::endl ;
    3092     LOG(LEVEL) << src->descELV(elv) ;
    3093     LOG(LEVEL) << src->descELV2(elv) ;
    3094 
    3095     CSGFoundry* dst = CSGCopy::Select(src, elv );
    3096     dst->setOrigin(src);
    3097     dst->setElv(elv);
    3098     dst->setOverrideSim(src->sim);
    3099 
    3100     LOG(LEVEL) << "]" ;
    3101     return dst ;
    3102 }
    3103 



HUH numSelectedPrim:2 when expected 1. CSGCopy=INFO cxr_min.sh
----------------------------------------------------------------


::

    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 0/10/   0 [R] : 5042:sWorld
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 1/10/   0 [F] : 5:PMT_3inch_pmt_solid
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 2/10/   0 [F] : 9:NNVTMCPPMTsMask_virtual
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 3/10/   0 [F] : 12:HamamatsuR12860sMask_virtual
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 4/10/   0 [F] : 4:mask_PMT_20inch_vetosMask_virtual
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 5/10/   0 [F] : 1:sStrutBallhead
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 6/10/   0 [F] : 1:base_steel
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 7/10/   0 [F] : 3:uni_acrylic1
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 8/10/   0 [F] : 130:sPanel
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copy@144]  sSolidIdx/sNumSolid/numSelectedPrim 9/10/   2 [T] : 338:ConnectingCutTube_0
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::copySolidInstances@421]  sNumInst 47888
    2025-09-30 09:59:18.069 INFO  [2916648] [CSGCopy::Select@55] 
    src:CSGFoundry  num_total 10 num_solid 10 num_prim 5545 num_node 29006 num_plan 0 num_tran 11610 num_itra 11610 num_inst 47888 gas 0 meshname 340 mmlabel 10 mtime 1759156584 mtimestamp 20250929_223624 sim Y
    dst:CSGFoundry  num_total 1 num_solid 1 num_prim 2 num_node 24 num_plan 0 num_tran 22 num_itra 22 num_inst 1 gas 0 meshname 340 mmlabel 1 mtime 1759156584 mtimestamp 20250929_223624 sim Y

    2025-09-30 09:59:18.071 INFO  [2916648] [CSGFoundry::getFrameE@3665]  MOI Waterdistributor_0:0:-2
    2025-09-30 09:59:18.072 INFO  [2916648] [CSGFoundry::getFrame@3519] [CSGFoundry__getFrame_VERBOSE] YES frs Waterdistributor_0:0:-2 looks_like_moi YES looks_like_raw NO 
    2025-09-30 09:59:18.072 INFO  [2916648] [CSGFoundry::getFrame@3538] [CSGFoundry__getFrame_VERBOSE] YES frs Waterdistributor_0:0:-2 looks_like_moi YES midx 37 mord 0 gord -2 rc 0
    2025-09-30 09:59:18.072 INFO  [2916648] [CSGFoundry::getFrame@3548] [CSGFoundry__getFrame_VERBOSE] YES[fr.desc


::

    117 void CSGCopy::copy()
    118 {
    119     CSGFoundry::CopyNames(dst, src );
    120     // No accounting for selection changing the number of names
    121     // as these are LV level names. The LV idx are regarded as 
    122     // external and unchanged no matter the selection, unlike gas_idx. 
    123 
    124     for(unsigned i=0 ; i < sNumSolid ; i++)
    125     {
    126         sSolidIdx = i ;
    127         solidMap[sSolidIdx] = -1 ;
    128 
    129         unsigned dump_ = Dump(sSolidIdx);
    130         bool dump_solid = dump_ & 0x1 ;
    131         LOG_IF(info, dump_solid)
    132             << "sSolidIdx " << sSolidIdx
    133             << " DUMP_RIDX " << DUMP_RIDX
    134             << " DUMP_NPS " << DUMP_NPS
    135             << " dump_solid " << dump_solid
    136             ;
    137 
    138         const CSGSolid* sso = src->getSolid(sSolidIdx);
    139         unsigned numSelectedPrim = src->getNumSelectedPrimInSolid(sso, elv );
    140         const std::string& solidMMLabel = src->getSolidMMLabel(sSolidIdx);
    141 
    142         char sIntent = sso->getIntent();
    143 
    144         LOG(LEVEL)
    145             << " sSolidIdx/sNumSolid/numSelectedPrim"
    146             << std::setw(2) << sSolidIdx
    147             << "/"
    148             << std::setw(2) << sNumSolid
    149             << "/"
    150             << std::setw(4) << numSelectedPrim
    151             << " [" << sIntent << "] "
    152             << ": "
    153             << solidMMLabel
    154             ;
    155 
    156         LOG_IF(LEVEL, dump_solid) << " sso " << sso->desc() << " numSelectedPrim " << numSelectedPrim << " solidMMLabel " << solidMMLabel ;
    157 
    158         if( numSelectedPrim == 0 ) continue ;
    159 
    160         dst->addSolidMMLabel( solidMMLabel.c_str() );
    161 
    162         unsigned dSolidIdx = dst->getNumSolid() ; // index before adding (0-based)
    163         if( elv == nullptr ) assert( dSolidIdx == sSolidIdx );
    164 
    165         CSGSolid* dso = dst->addSolid(numSelectedPrim, sso->label );
    166         int dPrimOffset = dso->primOffset ;




CSGFoundry::getNumSelectedPrimInSolid
----------------------------------------

Looks like have duplicated CSGPrim::meshIdx within the triangulated 'T' CSGSolid
that is actually in a different position.  That could be due to the
Peidong sharing G4VSolid between LV.

::

    2332 /**
    2333 CSGFoundry::getNumSelectedPrimInSolid
    2334 --------------------------------------
    2335 
    2336 Used by CSGCopy::copy
    2337 
    2338 Iterates over the CSGPrim within the CSGSolid counting the
    2339 number selected based on whether the CSGPrim::meshIdx
    2340 is within the elv SBitSet.
    2341 
    2342 
    2343 **/
    2344 
    2345 unsigned CSGFoundry::getNumSelectedPrimInSolid(const CSGSolid* solid, const SBitSet* elv ) const
    2346 {
    2347     unsigned num_selected_prim = 0 ;
    2348     for(int primIdx=solid->primOffset ; primIdx < solid->primOffset+solid->numPrim ; primIdx++)
    2349     {
    2350         const CSGPrim* pr = getPrim(primIdx);
    2351         unsigned meshIdx = pr->meshIdx() ;
    2352         bool selected = elv == nullptr ? true : elv->is_set(meshIdx) ;
    2353         num_selected_prim += int(selected) ;
    2354     }
    2355     return num_selected_prim ;
    2356 }
    2357 


::

    (ok) A[blyth@localhost CSGFoundry]$ cat.py primname.txt | grep Waterdistributor
    5211 Waterdistributor_1
    5212 Waterdistributor_0
    5213 Waterdistributor_1
    5214 Waterdistributor_0
    5219 Waterdistributor_3
    5220 Waterdistributor_2
    5221 Waterdistributor_3
    5222 Waterdistributor_2



::

    (ok) A[blyth@localhost CSGFoundry]$ cat.py primname.txt | more

    5206 sBar_0
    5207 ConnectingCutTube_0
    5208 ConnectingCutTube_1
    5209 ConnectingCutTube_2
    5210 WaterDistributorPartIIIUnion
    5211 Waterdistributor_1
    5212 Waterdistributor_0
    5213 Waterdistributor_1
    5214 Waterdistributor_0
    5215 sOuterReflectorInCD_TSubWaterDistributor_cutTube2
    5216 sOuterWaterInCD_TSubWaterDistributor_cutTube2
    5217 sInnerReflectorInCD_TSubWaterDistributor_cutTube2
    5218 sInnerWater_TSubWaterDistributor_cutTube2
    5219 Waterdistributor_3
    5220 Waterdistributor_2
    5221 Waterdistributor_3
    5222 Waterdistributor_2
    5223 solidSJCLSanchor
    5224 solidSJCLSanchor
    5225 solidSJFixture
    5226 solidSJFixture






