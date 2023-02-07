FIXED : CSGFoundry_stree_python_dumping_showing_some_typecode_CSG_TREE_where_CSG_UNION_expected.rst
======================================================================================================

Whilst comparing geometry details from CSG/CSGFoundry.py and sysrap/stree.p descSolids
find unexpected CSG_TREE. 

This is explained by a failure to update OpticksCSG.py following a change to OpticksCSG.h.
Added note to sysrap/CMakeLists.txt::

     30 OpticksCSG.py managed as source despite needing manual regeneration after OpticksCSG.h updates
     31 -------------------------------------------------------------------------------------------------
     32 
     33 NB generation of OpticksCSG.py is not automated, unlike the generation of other things like status json.  
     34 This is because it is expedient for the generated python to be managed as a source file for simplicity.  
     35 It would be cleaner to generate json that a future OpticksCSG.py parses. 
     36 But as OpticksCSG.h updates are rare, For now just need to remember to generate manually with::
     37 
     38     cd ~/opticks/sysrap
     39     ../bin/c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 
     40    


CSGFoundry::

    epsilon:tests blyth$ ./CSGFoundryLoadTest.sh ana


    CSGFoundry.descSolid ridx  2 label               r2 numPrim     11 primOffset   3094 
     px 3094 nn    7 no 23214 lv 128 pxl   0 :                            NNVTMCPPMTsMask_virtual : 1:tree 1:tree 108:cone 105:cylinder 105:cylinder 0:zero 0:zero 
     px 3095 nn    7 no 23221 lv 118 pxl   1 :                                    NNVTMCPPMTsMask : 2:union 1:tree 2:union 103:zsphere 105:cylinder 103:zsphere 105:cylinder 
     px 3096 nn   15 no 23228 lv 119 pxl   2 :                                     NNVTMCPPMTTail : 2:union 1:tree 2:union 1:tree 105:cylinder 2:union 105:cylinder 103:zsphere 105:cylinder 0:zero 0:zero 103:zsphere 105:cylinder 0:zero 0:zero 
     px 3097 nn    1 no 23243 lv 127 pxl   3 :               NNVTMCPPMT_PMT_20inch_pmt_solid_head : 103:zsphere 
     px 3098 nn    1 no 23244 lv 126 pxl   4 :              NNVTMCPPMT_PMT_20inch_body_solid_head : 103:zsphere 
     px 3099 nn    1 no 23245 lv 120 pxl   5 :            NNVTMCPPMT_PMT_20inch_inner1_solid_head : 103:zsphere 
     px 3100 nn    1 no 23246 lv 125 pxl   6 :            NNVTMCPPMT_PMT_20inch_inner2_solid_head : 103:zsphere 
     px 3101 nn    3 no 23247 lv 121 pxl   7 :                   NNVTMCPPMT_PMT_20inch_edge_solid : 2:union 105:cylinder 105:cylinder 
     px 3102 nn    3 no 23250 lv 122 pxl   8 :                  NNVTMCPPMT_PMT_20inch_plate_solid : 2:union 105:cylinder 105:cylinder 
     px 3103 nn    3 no 23253 lv 123 pxl   9 :                   NNVTMCPPMT_PMT_20inch_tube_solid : 2:union 105:cylinder 105:cylinder 
     px 3104 nn    1 no 23256 lv 124 pxl  10 :                    NNVTMCPPMT_PMT_20inch_mcp_solid : 105:cylinder 



epsilon:tests blyth$ GEOM=J007 ./stree_load_test.sh ana::

    stree.descSolid ridx   2 numPrim    11 lvid [128 118 119 127 126 120 125 121 122 123 124] n_lvid_one 1

     lv:128 nlv: 1                            NNVTMCPPMTsMask_virtual csg  4 tcn 105:cylinder 105:cylinder 108:cone 11:contiguous 
     lv:118 nlv: 1                                    NNVTMCPPMTsMask csg  7 tcn 103:zsphere 105:cylinder 1:tree 103:zsphere 105:cylinder 1:tree 3:intersection 
     lv:119 nlv: 1                                     NNVTMCPPMTTail csg 11 tcn 103:zsphere 105:cylinder 1:tree 105:cylinder 1:tree 103:zsphere 105:cylinder 1:tree 105:cylinder 1:tree 3:intersection 
     lv:127 nlv: 1               NNVTMCPPMT_PMT_20inch_pmt_solid_head csg  1 tcn 103:zsphere 
     lv:126 nlv: 1              NNVTMCPPMT_PMT_20inch_body_solid_head csg  1 tcn 103:zsphere 
     lv:120 nlv: 1            NNVTMCPPMT_PMT_20inch_inner1_solid_head csg  1 tcn 103:zsphere 
     lv:125 nlv: 1            NNVTMCPPMT_PMT_20inch_inner2_solid_head csg  1 tcn 103:zsphere 
     lv:121 nlv: 1                   NNVTMCPPMT_PMT_20inch_edge_solid csg  3 tcn 105:cylinder 105:cylinder 3:intersection 
     lv:122 nlv: 1                  NNVTMCPPMT_PMT_20inch_plate_solid csg  3 tcn 105:cylinder 105:cylinder 3:intersection 
     lv:123 nlv: 1                   NNVTMCPPMT_PMT_20inch_tube_solid csg  3 tcn 105:cylinder 105:cylinder 3:intersection 
     lv:124 nlv: 1                    NNVTMCPPMT_PMT_20inch_mcp_solid csg  1 tcn 105:cylinder 

     lv:128 nlv: 1                            NNVTMCPPMTsMask_virtual csg  4 tcn 105:cylinder 105:cylinder 108:cone 11:contiguous 
    desc_csg lvid:128 st.f.soname[128]:NNVTMCPPMTsMask_virtual 
            ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf
    array([[575,   1,   0, 578,   0,  -1, 576, 128, 105, 349, 349,  -1,   0,   0,   0,   0],
           [576,   1,   1, 578,   0,  -1, 577, 128, 105, 350, 350,  -1,   0,   0,   0,   0],
           [577,   1,   2, 578,   0,  -1,  -1, 128, 108, 351, 351,  -1,   0,   0,   0,   0],
           [578,   0,  -1,  -1,   3, 575,  -1, 128,  11,  -1,  -1,  -1,   0,   0,   0,   0]], dtype=int32)




::

    epsilon:tests blyth$ opticks-f CSG_TREE
    ./sysrap/tests/OpticksCSGTest.cc:            CSG_TREE, 
    ./sysrap/OpticksCSG.h:    CSG_TREE=1,
    ./sysrap/OpticksCSG.h://static const char* CSG_TREE_          = "tree" ; 
    ./sysrap/OpticksCSG.h:            //case CSG_TREE:          s = CSG_TREE_          ; break ;   CSG_TREE has same value as CSG_UNION it is used for grouping 
    ./npy/NNode.cpp:unsigned nnode::get_tree_mask() const { return get_mask(CSG_TREE) ; }  // formerly get_oper_mask
    ./npy/NNode.cpp:std::string nnode::get_tree_mask_string() const { return get_mask_string(CSG_TREE) ; }   // _oper
    ./npy/NNode.cpp:       else if(ntyp == CSG_TREE && CSG::IsTree(node->type)) collect = true ; 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 


AHHA : CSG_TREE has same enum value as CSG_UNION::


     27     CSG_TREE=1,
     28         CSG_UNION=1,
     29         CSG_INTERSECTION=2,
     30         CSG_DIFFERENCE=3,
     31 




After updating OpticksCSG.py the unexpected "tree" no longer appears::

    stree.descSolid ridx   2 numPrim    11 lvid [128 118 119 127 126 120 125 121 122 123 124] n_lvid_one 1

     lv:128 nlv: 1                            NNVTMCPPMTsMask_virtual csg  4 tcn 105:cylinder 105:cylinder 108:cone 11:contiguous 
     lv:118 nlv: 1                                    NNVTMCPPMTsMask csg  7 tcn 103:zsphere 105:cylinder 1:union 103:zsphere 105:cylinder 1:union 3:difference 
     lv:119 nlv: 1                                     NNVTMCPPMTTail csg 11 tcn 103:zsphere 105:cylinder 1:union 105:cylinder 1:union 103:zsphere 105:cylinder 1:union 105:cylinder 1:union 3:difference 
     lv:127 nlv: 1               NNVTMCPPMT_PMT_20inch_pmt_solid_head csg  1 tcn 103:zsphere 
     lv:126 nlv: 1              NNVTMCPPMT_PMT_20inch_body_solid_head csg  1 tcn 103:zsphere 
     lv:120 nlv: 1            NNVTMCPPMT_PMT_20inch_inner1_solid_head csg  1 tcn 103:zsphere 
     lv:125 nlv: 1            NNVTMCPPMT_PMT_20inch_inner2_solid_head csg  1 tcn 103:zsphere 
     lv:121 nlv: 1                   NNVTMCPPMT_PMT_20inch_edge_solid csg  3 tcn 105:cylinder 105:cylinder 3:difference 
     lv:122 nlv: 1                  NNVTMCPPMT_PMT_20inch_plate_solid csg  3 tcn 105:cylinder 105:cylinder 3:difference 
     lv:123 nlv: 1                   NNVTMCPPMT_PMT_20inch_tube_solid csg  3 tcn 105:cylinder 105:cylinder 3:difference 
     lv:124 nlv: 1                    NNVTMCPPMT_PMT_20inch_mcp_solid csg  1 tcn 105:cylinder 


    CSGFoundry.descSolid ridx  2 label               r2 numPrim     11 primOffset   3094 
     px 3094 nn    7 no 23214 lv 128 pxl   0 :                            NNVTMCPPMTsMask_virtual : 1:union 1:union 108:cone 105:cylinder 105:cylinder 0:zero 0:zero 
     px 3095 nn    7 no 23221 lv 118 pxl   1 :                                    NNVTMCPPMTsMask : 2:intersection 1:union 2:intersection 103:zsphere 105:cylinder 103:zsphere 105:cylinder 
     px 3096 nn   15 no 23228 lv 119 pxl   2 :                                     NNVTMCPPMTTail : 2:intersection 1:union 2:intersection 1:union 105:cylinder 2:intersection 105:cylinder 103:zsphere 105:cylinder 0:zero 0:zero 103:zsphere 105:cylinder 0:zero 0:zero 
     px 3097 nn    1 no 23243 lv 127 pxl   3 :               NNVTMCPPMT_PMT_20inch_pmt_solid_head : 103:zsphere 
     px 3098 nn    1 no 23244 lv 126 pxl   4 :              NNVTMCPPMT_PMT_20inch_body_solid_head : 103:zsphere 
     px 3099 nn    1 no 23245 lv 120 pxl   5 :            NNVTMCPPMT_PMT_20inch_inner1_solid_head : 103:zsphere 
     px 3100 nn    1 no 23246 lv 125 pxl   6 :            NNVTMCPPMT_PMT_20inch_inner2_solid_head : 103:zsphere 
     px 3101 nn    3 no 23247 lv 121 pxl   7 :                   NNVTMCPPMT_PMT_20inch_edge_solid : 2:intersection 105:cylinder 105:cylinder 
     px 3102 nn    3 no 23250 lv 122 pxl   8 :                  NNVTMCPPMT_PMT_20inch_plate_solid : 2:intersection 105:cylinder 105:cylinder 
     px 3103 nn    3 no 23253 lv 123 pxl   9 :                   NNVTMCPPMT_PMT_20inch_tube_solid : 2:intersection 105:cylinder 105:cylinder 
     px 3104 nn    1 no 23256 lv 124 pxl  10 :                    NNVTMCPPMT_PMT_20inch_mcp_solid : 105:cylinder 




The CSG differences remaining are more expected:

1. NNode has done positiv-ization (so get CSGFoundry sees "intersection" where stree sees "difference")

   * TODO: include complemented or not in outputs 

2. NNode has done complete-binary-tree-ization (so CSGFoundary sees lots of zero nodes)



