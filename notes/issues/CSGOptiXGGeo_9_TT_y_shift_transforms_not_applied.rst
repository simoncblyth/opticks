CSGOptiXGGeo_9_TT_y_shift_transforms_not_applied
==================================================

* https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210426.html
* https://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210504.html


Overview
--------------

This was caused by a fundamental difference between the CSGFoundry approach which 
uses global transform referencing and the old pre-7 approach that 
treats each compound solid in its own optix 6 geometry context.

The solution is to use **--gparts_transform_offset** when creating 
the CSGFoundry geometry::


    epsilon:CSG_GGeo blyth$ pwd
    /Users/blyth/CSG_GGeo
    epsilon:CSG_GGeo blyth$ cat run1.sh 
    #!/bin/bash -l
        
    export ONE_PRIM_SOLID=1,2,3,4
    #export ONE_NODE_SOLID=1,2,3,4,8
    #export DEEP_COPY_SOLID=1,2,3,4
    #export KLUDGE_SCALE_PRIM_BBOX=d1    # d1,d2

    ./run.sh --gparts_transform_offset 

     

GParts::add added transform offsets 
---------------------------------------


::

    1262 void GParts::add(GParts* other)
    1263 {
    1264     m_subs.push_back(other);
    1265 
    1266     if(getBndLib() == NULL)
    1267     {
    1268         setBndLib(other->getBndLib());
    1269     }
    1270     else
    1271     {
    1272         assert(getBndLib() == other->getBndLib());
    1273     }
    1274 
    1275     unsigned int n0 = getNumParts(); // before adding
    1276 
    1277     m_bndspec->add(other->getBndSpec());
    1278 
    1279 
    1280     // count the tran and plan collected so far into this GParts
    1281     unsigned tranOffset = m_tran_buffer->getNumItems();
    1282     //unsigned planOffset = m_plan_buffer->getNumItems(); 
    1283 
    1284     NPY<unsigned>* other_idx_buffer = other->getIdxBuffer() ;
    1285     NPY<float>* other_part_buffer = other->getPartBuffer()->clone() ;
    1286     NPY<float>* other_tran_buffer = other->getTranBuffer() ;
    1287     NPY<float>* other_plan_buffer = other->getPlanBuffer() ;
    1288 
    1289     bool preserve_zero = true ;
    1290     bool preserve_signbit = true ;
    1291     other_part_buffer->addOffset(GTRANSFORM_J, GTRANSFORM_K, tranOffset, preserve_zero, preserve_signbit );
    1292     // hmm offsetting of planes needs to be done only for parts of type CSG_CONVEXPOLYHEDRON 
    1293 
    1294     m_idx_buffer->add(other_idx_buffer);
    1295     m_part_buffer->add(other_part_buffer);
    1296     m_tran_buffer->add(other_tran_buffer);
    1297     m_plan_buffer->add(other_plan_buffer);
    1298     
    1299     unsigned num_idx_add = other_idx_buffer->getNumItems() ;
    1300     assert( num_idx_add == 1);
    1301     


The offsetting needs to be configurable as it is only needed 
for global handling of transforms, and would mess up the old geometry::


    1296 
    1297     if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    1298     {
    1299         LOG(LEVEL) << " --gparts_transform_offset " ;
    1300         bool preserve_zero = true ;
    1301         bool preserve_signbit = true ;
    1302         other_part_buffer->addOffset(GTRANSFORM_J, GTRANSFORM_K, tranOffset, preserve_zero, preserve_signbit );
    1303         // hmm offsetting of planes needs to be done only for parts of type CSG_CONVEXPOLYHEDRON 
    1304     }
    1305     else
    1306     {
    1307         LOG(LEVEL) << " NOT --gparts_transform_offset " ;
    1308     }
    1309 
    1310     m_idx_buffer->add(other_idx_buffer);
    1311     m_part_buffer->add(other_part_buffer);
    1312     m_tran_buffer->add(other_tran_buffer);
    1313     m_plan_buffer->add(other_plan_buffer);





GParts::getGTransform comes from the m_part_buffer
------------------------------------------------------

* on combination the GTransform indices from the individual GParts 
  need to be offset according to the total number of transforms that 
  have been collected into the combined m_tran_buffer

  * THIS IS TRUE FOR GLOBAL TRANSFORM HANDLING, NOT PER-GEOMETRY HANDLING  

::


    1965 unsigned int GParts::getUInt(unsigned int i, unsigned int j, unsigned int k) const
    1966 {
    1967     assert(i < getNumParts() );
    1968     unsigned int l=0u ;
    1969     return m_part_buffer->getUInt(i,j,k,l);
    1970 }


    1986 unsigned GParts::getGTransform(unsigned partIdx) const
    1987 {
    1988     unsigned q3_u_w = getUInt(partIdx, GTRANSFORM_J, GTRANSFORM_K);
    1989     return q3_u_w & 0x7fffffff ;
    1990 }




Possibly the problem is not lack of tranforms, rather it is due to the GTransform not being offset with the combination
---------------------------------------------------------------------------------------------------------------------------

* so end up always using the first transform 
* THE PROBLEM WAS THE REFERENCING OF THE TRANSFORMS, NOT THE PRESENCE OF THEM


::

    189 CSGNode* Converter::convert_(const GParts* comp, unsigned primIdx, unsigned partIdxRel )
    190 {
    191     unsigned repeatIdx = comp->getRepeatIndex();  // set in GGeo::deferredCreateGParts
    192     unsigned partOffset = comp->getPartOffset(primIdx) ;
    193     unsigned partIdx = partOffset + partIdxRel ;
    194     unsigned idx = comp->getIndex(partIdx);
    195     assert( idx == partIdx );
    196 
    197     std::string tag = comp->getTag(partIdx);
    198     unsigned tc = comp->getTypeCode(partIdx);
    199 
    200     unsigned gtran = 0 ;
    201     const Tran<float>* tv = nullptr ;
    202 
    203     if( splay != 0.f )    // splaying currently prevents the real transform from being used 
    204     {
    205         tv = Tran<float>::make_translate(0.f, float(primIdx)*splay, float(partIdxRel)*splay );
    206     }
    207     else
    208     {
    209         gtran = comp->getGTransform(partIdx);
    210         if( gtran > 0 )
    211         {
    212             glm::mat4 t = comp->getTran(gtran-1,0) ;
    213             glm::mat4 v = comp->getTran(gtran-1,1);
    214             tv = new Tran<float>(t, v);
    215         }
    216     }




Huh the y-shifts are ine GParts com too
-----------------------------------------

::

    epsilon:CSGOptiXGGeo blyth$ GPARTS_DEBUG=9 ./CSGOptiXGGeo.sh 9d
    ...

    2021-05-05 15:04:08.799 INFO  [4771509] [GParts::applyPlacementTransform@1223]  num_mismatch 0
     mismatch indices : 
    2021-05-05 15:04:08.799 INFO  [4771509] [*GParts::Create@292]  parts.numTran 1 parts.getTran 0 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 831.600 0.000 1.000  
    2021-05-05 15:04:08.799 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 0 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 0.000 0.000 1.000  
    2021-05-05 15:04:08.799 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 1 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 0.000 0.000 1.000  
    2021-05-05 15:04:08.799 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 2 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -831.600 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 3 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -831.600 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 4 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -805.200 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 5 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -805.200 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 6 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -778.800 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 7 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -778.800 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 8 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -752.400 0.000 1.000  
    2021-05-05 15:04:08.800 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 9 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -752.400 0.000 1.000  

    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 121 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 726.000 0.000 1.000  
    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 122 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 752.400 0.000 1.000  
    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 123 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 752.400 0.000 1.000  
    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 124 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 778.800 0.000 1.000  
    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 125 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 778.800 0.000 1.000  
    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 126 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 805.200 0.000 1.000  
    2021-05-05 15:04:08.804 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 127 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 805.200 0.000 1.000  
    2021-05-05 15:04:08.805 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 128 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 831.600 0.000 1.000  
    2021-05-05 15:04:08.805 INFO  [4771509] [*GParts::Create@307]  com.numTran 130 com.getTran 129 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 831.600 0.000 1.000  
    2021-05-05 15:04:08.805 INFO  [4771509] [*GParts::Create@324] ]
    2021-05-05 15:04:08.805 INFO  [4771509] [GGeo::dumpParts@1455] CSGOptiXGGeo.main




GParts::applyPlacementTransform sees them too : applied to pre combination GParts:m_tran_buffer
--------------------------------------------------------------------------------------------------

::

    2021-05-05 14:50:00.223 INFO  [4755889] [GParts::applyPlacementTransform@1164]  tran_buffer 1,3,4,4 ni 1
    2021-05-05 14:50:00.223 INFO  [4755889] [nmat4triple::dump@300] GParts::applyPlacementTransform before
      0 tvq 
      triple.t identity 

      triple.v identity 

      triple.q identity 


    2021-05-05 14:50:00.223 INFO  [4755889] [nmat4triple::dump@300] GParts::applyPlacementTransform after
      0 tvq 
      triple.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000 541.200   0.000   1.000 

      triple.v  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000 -541.200   0.000   1.000 

      triple.q  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000 -541.200 
                0.000   0.000   1.000   0.000 
                0.000   0.000   0.000   1.000 





GParts::Create does see them too
------------------------------------


::

    epsilon:CSGOptiXGGeo blyth$ GPARTS_DEBUG=9 ./CSGOptiXGGeo.sh 9d
    repeatIdx 9
    2021-05-05 14:36:36.148 INFO  [4743592] [*GParts::Create@232] [  deferred creation from GPts
    2021-05-05 14:36:36.148 INFO  [4743592] [*GParts::Create@238]  num_pt 130
    2021-05-05 14:36:36.148 INFO  [4743592] [*GParts::Create@252]  pt  lv    7 nd     10 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 0.000 0.000 1.000   bn Air///Aluminium
    2021-05-05 14:36:36.148 INFO  [4743592] [*GParts::Create@252]  pt  lv    6 nd     11 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 0.000 0.000 1.000   bn Aluminium///Adhesive
    2021-05-05 14:36:36.149 INFO  [4743592] [*GParts::Create@252]  pt  lv    5 nd     12 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -831.600 0.000 1.000   bn Adhesive///TiO2Coating
    2021-05-05 14:36:36.149 INFO  [4743592] [*GParts::Create@252]  pt  lv    4 nd     13 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -831.600 0.000 1.000   bn TiO2Coating///Scintillator
    2021-05-05 14:36:36.149 INFO  [4743592] [*GParts::Create@252]  pt  lv    5 nd     14 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -805.200 0.000 1.000   bn Adhesive///TiO2Coating
    2021-05-05 14:36:36.149 INFO  [4743592] [*GParts::Create@252]  pt  lv    4 nd     15 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -805.200 0.000 1.000   bn TiO2Coating///Scintillator
    2021-05-05 14:36:36.149 INFO  [4743592] [*GParts::Create@252]  pt  lv    5 nd     16 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -778.800 0.000 1.000   bn Adhesive///TiO2Coating
    2021-05-05 14:36:36.149 INFO  [4743592] [*GParts::Create@252]  pt  lv    4 nd     17 pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -778.800 0.000 1.000   bn TiO2Coating///Scintillator


Maybe the placement action is in NCSG which is not being consulted by the CSGFoundry conversion::

     252         LOG(level)
     253             << " pt " << std::setw(4)
     254             << " lv " << std::setw(4) << lvIdx
     255             << " nd " << std::setw(6) << ndIdx
     256             << " pl " << GLMFormat::Format(placement)
     257             << " bn " << spec
     258             ;
     259 
     260         assert( lvIdx > -1 );
     261 
     262         const NCSG* csg = unsigned(lvIdx) < solids.size() ? solids[lvIdx] : NULL ;
     263         assert( csg );
     264 
     265         //  X4PhysicalVolume::convertNode
     266 
     267         GParts* parts = GParts::Make( csg, spec.c_str(), ndIdx );
     268 
     269         unsigned num_mismatch = 0 ;
     270 
     271         parts->applyPlacementTransform( placement, verbosity, num_mismatch );
     272 




y-shift placment transforms are there in GPt
----------------------------------------------

::

    epsilon:ggeo blyth$ GPtTest 9
    2021-05-05 14:00:05.106 INFO  [4704311] [main@64] 
     idpath  /usr/local/opticks/geocache/OKX4Test_lWorld0x33e33d0_PV_g4live/g4ok_gltf/e33b2270395532f5661fde4c61889844/1
     objpath /usr/local/opticks/geocache/OKX4Test_lWorld0x33e33d0_PV_g4live/g4ok_gltf/e33b2270395532f5661fde4c61889844/1/GPts/9

    2021-05-05 14:00:05.107 INFO  [4704311] [GPts::dump@201] GPts::dump GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
     i    0 lv   7 cs   7 nd      10 bn                Air///Aluminium pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 0.000 0.000 1.000  
     i    1 lv   6 cs   6 nd      11 bn           Aluminium///Adhesive pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 0.000 0.000 1.000  
     i    2 lv   5 cs   5 nd      12 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -831.600 0.000 1.000  
     i    3 lv   4 cs   4 nd      13 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -831.600 0.000 1.000  
     i    4 lv   5 cs   5 nd      14 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -805.200 0.000 1.000  
     i    5 lv   4 cs   4 nd      15 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -805.200 0.000 1.000  
     i    6 lv   5 cs   5 nd      16 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -778.800 0.000 1.000  
     i    7 lv   4 cs   4 nd      17 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -778.800 0.000 1.000  
     i    8 lv   5 cs   5 nd      18 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -752.400 0.000 1.000  
     i    9 lv   4 cs   4 nd      19 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -752.400 0.000 1.000  
     i   10 lv   5 cs   5 nd      20 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -726.000 0.000 1.000  
     i   11 lv   4 cs   4 nd      21 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 -726.000 0.000 1.000  
    ...
     i  122 lv   5 cs   5 nd     132 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 752.400 0.000 1.000  
     i  123 lv   4 cs   4 nd     133 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 752.400 0.000 1.000  
     i  124 lv   5 cs   5 nd     134 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 778.800 0.000 1.000  
     i  125 lv   4 cs   4 nd     135 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 778.800 0.000 1.000  
     i  126 lv   5 cs   5 nd     136 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 805.200 0.000 1.000  
     i  127 lv   4 cs   4 nd     137 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 805.200 0.000 1.000  
     i  128 lv   5 cs   5 nd     138 bn         Adhesive///TiO2Coating pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 831.600 0.000 1.000  
     i  129 lv   4 cs   4 nd     139 bn     TiO2Coating///Scintillator pl 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0  0.000 831.600 0.000 1.000  
    epsilon:ggeo blyth$ 


mm:9 many bbox on top of each other
-----------------------------------------

::

    primIdx 121 CSGPrim mn (-3430.0, -13.0,  -5.0)  mx (3430.0,  13.0,   5.0)  numNode/sbt/node/tran/planOffset    1 121 121   0   0
     primIdx 122 CSGPrim mn (-3430.0, -13.1,  -5.2)  mx (3430.0,  13.1,   5.2)  numNode/sbt/node/tran/planOffset    1 122 122   0   0
     primIdx 123 CSGPrim mn (-3430.0, -13.0,  -5.0)  mx (3430.0,  13.0,   5.0)  numNode/sbt/node/tran/planOffset    1 123 123   0   0
     primIdx 124 CSGPrim mn (-3430.0, -13.1,  -5.2)  mx (3430.0,  13.1,   5.2)  numNode/sbt/node/tran/planOffset    1 124 124   0   0
     primIdx 125 CSGPrim mn (-3430.0, -13.0,  -5.0)  mx (3430.0,  13.0,   5.0)  numNode/sbt/node/tran/planOffset    1 125 125   0   0
     primIdx 126 CSGPrim mn (-3430.0, -13.1,  -5.2)  mx (3430.0,  13.1,   5.2)  numNode/sbt/node/tran/planOffset    1 126 126   0   0
     primIdx 127 CSGPrim mn (-3430.0, -13.0,  -5.0)  mx (3430.0,  13.0,   5.0)  numNode/sbt/node/tran/planOffset    1 127 127   0   0
     primIdx 128 CSGPrim mn (-3430.0, -13.1,  -5.2)  mx (3430.0,  13.1,   5.2)  numNode/sbt/node/tran/planOffset    1 128 128   0   0
     primIdx 129 CSGPrim mn (-3430.0, -13.0,  -5.0)  mx (3430.0,  13.0,   5.0)  numNode/sbt/node/tran/planOffset    1 129 129   0   0
    CSGSolid  r009 primNum/Offset 130  0 ce (   0.0,   0.0,   0.0,3430.6) 
    CSGNode     0 bo aabb: -3430.6  -846.2    -6.7  3430.6   846.2     6.7 
    CSGNode     1 bo aabb: -3430.0  -845.7    -6.1  3430.0   845.7     6.1 
    CSGNode     2 bo aabb: -3430.0   -13.1    -5.2  3430.0    13.1     5.2 
    CSGNode     3 bo aabb: -3430.0   -13.0    -5.0  3430.0    13.0     5.0 
    CSGNode     4 bo aabb: -3430.0   -13.1    -5.2  3430.0    13.1     5.2 
    CSGNode     5 bo aabb: -3430.0   -13.0    -5.0  3430.0    13.0     5.0 
    CSGNode     6 bo aabb: -3430.0   -13.1    -5.2  3430.0    13.1     5.2 
    CSGNode     7 bo aabb: -3430.0   -13.0    -5.0  3430.0    13.0     5.0 
    CSGNode     8 bo aabb: -3430.0   -13.1    -5.2  3430.0    13.1     5.2 
    CSGNode     9 bo aabb: -3430.0   -13.0    -5.0  3430.0    13.0     5.0 
    CSGNode    10 bo aabb: -3430.0   -13.1    -5.2  3430.0    13.1     5.2 
    CSGNode    11 bo aabb: -3430.0   -13.0    -5.0  3430.0    13.0     5.0 
    CSGNode    12 bo aabb: -3430.0   -13.1    -5.2  3430.0    13.1     5.2 
    CSGNode    13 bo aabb: -3430.0   -13.0    -5.0  3430.0    13.0     5.0 




    In [9]: np.set_printoptions(edgeitems=200)
    In [10]: a.reshape(-1,16)
    Out[10]:
    array([[   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -831.6,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -831.6,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -805.2,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -805.2,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -778.8,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -778.8,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -752.4,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -752.4,    0. ,    1. ],


    In [13]: a[:,3,1].reshape(-1,2)
    Out[13]:
    array([[   0. ,    0. ],
           [-831.6, -831.6],
           [-805.2, -805.2],
           [-778.8, -778.8],
           [-752.4, -752.4],
           [-726. , -726. ],
           [-699.6, -699.6],
           [-673.2, -673.2],
           [-646.8, -646.8],
           ...

    In [14]: a[:,3,1].reshape(-1,2).shape
    Out[14]: (65, 2)


    In [16]: a[:,3,1][::2]
    Out[16]:
    array([   0. , -831.6, -805.2, -778.8, -752.4, -726. , -699.6, -673.2, -646.8, -620.4, -594. , -567.6, -541.2, -514.8, -488.4, -462. , -435.6, -409.2, -382.8, -356.4, -330. , -303.6, -277.2, -250.8,
           -224.4, -198. , -171.6, -145.2, -118.8,  -92.4,  -66. ,  -39.6,  -13.2,   13.2,   39.6,   66. ,   92.4,  118.8,  145.2,  171.6,  198. ,  224.4,  250.8,  277.2,  303.6,  330. ,  356.4,  382.8,
            409.2,  435.6,  462. ,  488.4,  514.8,  541.2,  567.6,  594. ,  620.4,  646.8,  673.2,  699.6,  726. ,  752.4,  778.8,  805.2,  831.6], dtype=float32)

    In [17]: np.diff( a[:,3,1][::2] )
    Out[17]:
    array([-831.6,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,
             26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,
             26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4,   26.4], dtype=float32)

    In [18]:

