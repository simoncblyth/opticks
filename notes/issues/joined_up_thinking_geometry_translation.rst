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




Serialize n-ary tree
-----------------------

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

 

