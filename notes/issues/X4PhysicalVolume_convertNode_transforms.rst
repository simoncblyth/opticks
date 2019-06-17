X4PhysicalVolume_convertNode_transforms
===========================================

Context
----------

* :doc:`GPts_GParts_optimization`

Moving to deferred GParts creation, shaves ~5s and 1.3G from X4PhysicalVolume::convertStructure::

     diffListedTime           Time      DeltaTime             VM        DeltaVM
              0.000           0.000      16725.602          0.000        484.312 : OpticksRun::OpticksRun_1139392501
              0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
              0.012           0.012          0.012        103.880        103.880 : _OKX4Test:GGeo_0
              0.006           0.018          0.006        103.880          0.000 : OKX4Test:GGeo_0
              0.000           0.018          0.000        103.880          0.000 : _OKX4Test:X4PhysicalVolume_0
              0.000           0.018          0.000        103.880          0.000 : _X4PhysicalVolume::convertMaterials_0
              0.002           0.020          0.002        104.012          0.132 : X4PhysicalVolume::convertMaterials_0
              0.057           0.076          0.057        104.276          0.264 : _X4PhysicalVolume::convertSolids_0
              1.037           1.113          1.037        116.792         12.516 : X4PhysicalVolume::convertSolids_0
              0.000           1.113          0.000        116.792          0.000 : _X4PhysicalVolume::convertStructure_0
             21.137          22.250         21.137       2442.088       2325.296 : X4PhysicalVolume::convertStructure_0
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              0.000          22.250          0.000       2442.088          0.000 : OKX4Test:X4PhysicalVolume_0
              0.002          22.252          0.002       2442.088          0.000 : GInstancer::createInstancedMergedMeshes_0
              1.678          23.930          1.678       2630.452        188.364 : GInstancer::createInstancedMergedMeshes:deltacheck_0
             12.887          36.816         12.887       2999.904        369.452 : GInstancer::createInstancedMergedMeshes:traverse_0
              0.514          37.330          0.514       2999.904          0.000 : GInstancer::createInstancedMergedMeshes:labelTree_0
              0.000          37.330          0.000       2999.904          0.000 : _GMergedMesh::Create_0
              0.109          37.439          0.109       2999.904          0.000 : GMergedMesh::Create::Count_0
              0.000          37.439          0.000       2999.904          0.000 : _GMergedMesh::Create::Allocate_0
              0.021          37.461          0.021       3049.668         49.764 : GMergedMesh::Create::Allocate_0
             13.561          51.021         13.561       4668.632       1618.964 : GMergedMesh::Create::Merge_0
              0.002          51.023          0.002       4668.796          0.164 : GMergedMesh::Create::Bounds_0
             ....
              0.000          52.254          0.000       4697.472          0.000 : GMergedMesh::Create::Bounds_0
              0.076          52.330          0.076       4697.472          0.000 : GInstancer::createInstancedMergedMeshes:makeMergedMeshAndInstancedBuffers_0
              0.119          52.449          0.119       4697.472          0.000 : _OKX4Test:OKMgr_0
              4.029          56.479          4.029      13938.203       9240.731 : OKX4Test:OKMgr_0


TODO
--------

* investigate the culprits and try some alternatives 

::

    npy/NXform.hpp
    npy/NGLMExt.cpp





It would save some time if can use preexisting inverse from G4 ?
----------------------------------------------------------------------

* started looking into this behind X4_TRANSFORM macro


Currently using X4Transform3D::GetObjectTransform and are then inverting it.


g4-cls G4VPhysicalVolume::

    110     // Access functions
    111     //
    112     // The following are accessor functions that make a distinction
    113     // between whether the rotation/translation is being made for the
    114     // frame or the object/volume that is being placed.
    115     // (They are the inverse of each other).
    116 
    117     G4RotationMatrix* GetObjectRotation() const;              //  Obsolete 
    118     G4RotationMatrix  GetObjectRotationValue() const;  //  Replacement
    119     G4ThreeVector  GetObjectTranslation() const;
    120       // Return the rotation/translation of the Object relative to the mother.
    121     const G4RotationMatrix* GetFrameRotation() const;
    122     G4ThreeVector  GetFrameTranslation() const;
    123       // Return the rotation/translation of the Frame used to position 
    124       // this volume in its mother volume (opposite of object rot/trans).
    125 


om-cls X4Transform3D::

     17 glm::mat4 X4Transform3D::GetObjectTransform(const G4VPhysicalVolume* const pv)
     18 {
     19    // preferred for interop with glm/Opticks : obj relative to mother
     20     G4RotationMatrix rot = pv->GetObjectRotationValue() ; 
     21     G4ThreeVector    tla = pv->GetObjectTranslation() ;
     22     G4Transform3D    tra(rot,tla);
     23     return Convert( tra ) ;
     24 }
     25 
     26 glm::mat4 X4Transform3D::GetFrameTransform(const G4VPhysicalVolume* const pv)
     27 {
     28     const G4RotationMatrix* rotp = pv->GetFrameRotation() ;
     29     G4ThreeVector    tla = pv->GetFrameTranslation() ;
     30     G4Transform3D    tra(rotp ? *rotp : G4RotationMatrix(),tla);
     31     return Convert( tra ) ;
     32 }








Issue : transform handling takes 75% of convertNodes time
-------------------------------------------------------------

Around 15s of the 21s are doing transform handling::

    1072  m_convertNode_boundary_dt 3.47852
    1073  m_convertNode_transformsA_dt 0.644531
    1074  m_convertNode_transformsB_dt 6.35547
    1075  m_convertNode_transformsC_dt 0.277344
    1076  m_convertNode_transformsD_dt 7.29492
    1077  m_convertNode_transformsE_dt 0.230469
    1078  m_convertNode_GVolume_dt 3
 

Identified the hottest of the hot node code::


    1043      ///////////////////////////////////////////////////////////////  
    1044 
    1045 #ifdef X4_PROFILE
    1046     float t10 = BTimeStamp::RealTime();
    1047 #endif
    1048 
    1049     GPt* pt = new GPt( lvIdx, ndIdx, csgIdx, boundaryName.c_str() )  ;
    1050 
    1051     glm::mat4 xf_local = X4Transform3D::GetObjectTransform(pv);
    1052 
    1053 #ifdef X4_PROFILE
    1054     float t12 = BTimeStamp::RealTime();
    1055 #endif
    1056 
    1057     const nmat4triple* ltriple = m_xform->make_triple( glm::value_ptr(xf_local) ) ;   // YIKES does polardecomposition + inversion and checks them 
    1058     // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    1059 
    1060 #ifdef X4_PROFILE
    1061     float t13 = BTimeStamp::RealTime();
    1062 #endif
    1063 
    1064     GMatrixF* ltransform = new GMatrix<float>(glm::value_ptr(xf_local));
    1065 
    1066 #ifdef X4_PROFILE
    1067     float t15 = BTimeStamp::RealTime();
    1068 #endif
    1069 
    1070     X4Nd* nd = new X4Nd { parent_nd, ltriple } ;
    1071 
    1072     const nmat4triple* gtriple = nxform<X4Nd>::make_global_transform(nd) ;
    1073     // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            .. collects pointers to triples up the tree into a vector and then multiplies them ...
            .. doing both and t and v of the tvq ...

    1074 
    1075 #ifdef X4_PROFILE
    1076     float t17 = BTimeStamp::RealTime();
    1077 #endif
    1078 
    1079     glm::mat4 xf_global = gtriple->t ;
    1080 
    1081     GMatrixF* gtransform = new GMatrix<float>(glm::value_ptr(xf_global));
    1082 
    1083 #ifdef X4_PROFILE
    1084     float t20 = BTimeStamp::RealTime();
    1085 
    1086     m_convertNode_boundary_dt    += t10 - t00 ;
    1087 
    1088     m_convertNode_transformsA_dt += t12 - t10 ;
    1089     m_convertNode_transformsB_dt += t13 - t12 ;
    1090     m_convertNode_transformsC_dt += t15 - t13 ;
    1091     m_convertNode_transformsD_dt += t17 - t15 ;
    1092     m_convertNode_transformsE_dt += t20 - t17 ;
    1093 #endif
    1094 
    1095 /*
    1096      m_convertNode_boundary_dt 3.47852
    1097      m_convertNode_transformsA_dt 0.644531
    1098      m_convertNode_transformsB_dt 6.35547
    1099      m_convertNode_transformsC_dt 0.277344
    1100      m_convertNode_transformsD_dt 7.29492
    1101      m_convertNode_transformsE_dt 0.230469
    1102      m_convertNode_GVolume_dt 3
    1103 */
    1104 
    1105     ////////////////////////////////////////////////////////////////

