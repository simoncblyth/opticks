joined_up_thinking_geometry_translation
==========================================

Current geometry translation has historical baggage, and lots more code than necessary
as it has a complete geometry model in the middle::

    Geant4 -> GGeo/NPY  -> CSGFoundry 


How to handle double precision instance transforms in CSGFoundry ?
----------------------------------------------------------------------

* collect and persist in double precision
* narrow to float only transiently in memory just before uploading 
* so the float precision transforms only actually get used on device  
* dqat4 sibling of qat4 for use on host 


Explorations 
---------------

u4/U4Transform.h
    get object and frame transforms from PV 
    
u4/tests/U4TransformTest.cc
    test getting transforms from PV and writing them into NP arrays 
    or glm::tmat4x4<double> 

u4/U4Tree.h 
    thinking about minimal structure translation

    * is serialization of the intermediate tree needed ? 
    * probably not     

"SVolume"
    what is needed in the "SNode/SND" 
    HMM: actually due to old solid NNode is better to call "SVolume"

    Needs to mimic the GInstancer digest 

    * node_index 
    * glm::tmat4x4<double> obj
    * glm::tmat4x4<double> frm 
    * copyNo 

    Hmm parent and child indices easier to persist:

    * SVolume* parent     
    * std::vector<SVolume*> children 

    Serializing the below usefully needs indices
    associated with names.   

    * void* pv 
    * void* lv
    * void* so
    * void* mt




n-ary tree data structure
----------------------------

* https://www.enjoyalgorithms.com/blog/n-ary-tree

* convert n-ary to binary tree : by links to first child and next sibling 

Note every node is same size then::

    struct NaryTreeNode
    {
        int value ; 
        NaryTreeNode* firstChild ; 
        NaryTreeNode* nextSibling ; 
    };


What about::

    struct Node
    {
        "int value ;" 
        int numChild ;         // siblings stored contiguosly 
        int firstChildIndex ;  // -1 when no child 
        int parentIndex ;      // -1 for root 
    }; 
  
* keeping the "node header" to a max of 4 int32 will allow it to be squeezed into 4x4 transform spare 


Tree background
--------------------

* https://hbfs.wordpress.com/2009/04/07/compact-tree-storage/

* https://hbfs.wordpress.com/2016/09/06/serializing-trees/#1



TODO : compare stree_test with GGeo 
---------------------------------------

::

    [ stree::disqualifyContainedRepeats 
    ] stree::disqualifyContainedRepeats  disqualify.size 23
    [ stree::sortSubtrees 
    ] stree::sortSubtrees 
    st.desc_sub
        0 : c2520d0897b02efe301aed3f8d8b41e8 : 32256 de:( 9  9) 1st:    17 sBar0x71a9200
        1 : 246cf1cae2a2304dad8dbafa5238934f : 25600 de:( 6  6) 1st:194249 PMT_3inch_pmt_solid0x66e59f0
        2 : e238e3e830cc4e95eb9b167c54d155a2 : 12612 de:( 6  6) 1st: 70965 NNVTMCPPMTsMask_virtual0x5f5f900
        3 : 881ef0f2f7f79f81479dd6e0a07a380b :  5000 de:( 6  6) 1st: 70972 HamamatsuR12860sMask_virtual0x5f50d40
        4 : 25ed11817b62fa562aaef3daba337336 :  2400 de:( 4  4) 1st:322253 mask_PMT_20inch_vetosMask_virtual0x5f62e40
        5 : c051c1bb98b71ccb15b0cf9c67d143ee :   590 de:( 6  6) 1st: 68493 sStrutBallhead0x5853640
        6 : 5e01938acb3e0df0543697fc023bffb1 :   590 de:( 6  6) 1st: 69083 uni10x5832ff0
        7 : cdc824bf721df654130ed7447fb878ac :   590 de:( 6  6) 1st: 69673 base_steel0x58d3270
        8 : 3fd85f9ee7ca8882c8caa747d0eef0b3 :   590 de:( 6  6) 1st: 70263 uni_acrylic10x597c090
        9 : d4f5974d740cd7c78613c9d8563878c7 :   504 de:( 7  7) 1st:    15 sPanel0x71a8d90





DONE : Serialize n-ary tree
-----------------------------

* HMM by CSG list-nodes are related to this, should review them 

* https://www.geeksforgeeks.org/serialize-deserialize-n-ary-tree/

* :google:`tree serialization generic tree`

* https://eli.thegreenplace.net/2011/09/29/an-interesting-tree-serialization-algorithm-from-dwarf


Here's a quote from the DWARF v3 standard section 2.3 explaining it, slightly rephrased:

The tree itself is represented by flattening it in prefix order. Each node is
defined either to have children or not to have children. If a node is defined
not to have children, the next physically succeeding node is a sibling. If a
node is defined to have children, the next physically succeeding node is its
first child. Additional children are represented as siblings of the first
child. A chain of sibling entries is terminated by a null node.

 

