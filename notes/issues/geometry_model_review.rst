geometry_model_review
=======================

Objective of this review
--------------------------

High level description of the GGeo geometry model:
how it is constructed, issues with mm0.

See Also 
----------

* :doc:`identity_review.rst`
* :doc:`GGeo-OGeo-identity-direct-review.rst` Mechanics of the direct workflow:

Relevant Tests
----------------

GGeoConvertTest 
    does GGeo::dryrun_convert checking what OGeo::convert will do 
GGeoIdentityTest
    access identity across all mm 
GGeoTest 
    collective
ana/GNodeLib.py
    loads/dumps
OTracerTest
    just loads and visualizes geometry avoiding genstep issues
OKTest 
    full geom + genstep, without G4  


Simplifications to GMergedMesh
---------------------------------

* Can I get rid of top slot "globalinstance" with mm0 effectively becoming it ?


Summary of issue with Geometry Model
--------------------------------------

The root cause of the problems are trying to do too much in GMergedMesh slot 0 (aka mm0).

It tries to carry both:

1. all volume "global" information
2. non-instanced "remainder" information

It kinda gets away with this conflation by splitting on high-level/low-level axis using "selected" volumes.
But the result is still confusing even when it can be made to work, so it is prone to breakage.


globalinstance just adding to confusion
-------------------------------------------

In Aug I added an extra mm slot, called the GlobalInstance, which 
treats the remainder geometry just like instanced. That was motivated 
by identity access problems.

TODO: change the GlobalInstance -> RemainderInstance  

* BUT: This step might just be adding confusion. 
* MUST: make GMergedMesh simpler by doing less in it 

::

     415 void GMergedMesh::countVolume( GVolume* volume, bool selected, unsigned verbosity )
     416 {
     417     const GMesh* mesh = volume->getMesh();
     418 
     419     // with globalinstance selection is honoured at volume level too 
     420     bool admit = ( m_globalinstance && selected ) || !m_globalinstance ;  
     421     if(admit)
     422     {
     423         m_num_volumes += 1 ; 
     424     }
     425     if(selected)
     426     {
     427         m_num_volumes_selected += 1 ;
     428         countMesh( mesh ); 
     429     }   
     430     
     431     //  hmm having both admit and selected is confusing 
     432     


Idea for solution : keep the "all" info in GNodeLib arrays and get it persisted there 
---------------------------------------------------------------------------------------

Need to have access to both "all" geometry info and "remainder" geometry info.
Both these currently in GMergedMesh slot zero. 

Investigate the usage of the "all" info from mm0, 

1. relocating the "all volume info" collection into GNodeLib::

    void GNodeLib::add(const GVolume* volume)

2. get it persisted there (currently just persists pvlist lvlist names) for all volumes a


This would help by moving GMergedMesh in the simpler direction.



TODO: find "full volume" users of the old mm0/mesh0 and convert them to use GNodeLib
------------------------------------------------------------------------------------- 

::

    epsilon:opticks blyth$ opticks-fl mm0
    ./ana/flightpath.py
    ./ana/geom2d.py
    ./ana/view.py
    ./ana/mm0prim2.py
    ./ana/geocache.bash
    ./opticksgeo/OpticksAim.hh
    ./opticksgeo/OpticksHub.cc
    ./opticksgeo/OpticksAim.cc
    ./bin/ab.bash
    ./ok/ok.bash
    ./extg4/X4Transform3D.cc
    ./ggeo/GParts.cc
    ./ggeo/GGeoTest.hh
    ./ggeo/GMesh.cc
    ./ggeo/GGeo.cc
    ./ggeo/GScene.hh
    ./ggeo/GInstancer.cc
    ./ggeo/GGeoLib.cc
    ./ggeo/GMergedMesh.cc
    ./ggeo/GScene.cc
    ./optickscore/OpticksDomain.hh
    ./npy/NScene.cpp
    ./oglrap/Scene.cc




STEPS TO MINIMIZE DUPLICATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. simplify GMergedMesh using self contained code pull offs into other classes, eg GVolume


BUT : do not want to duplicate code in GMergedMesh and GNodeLib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* move GVolume->array content processing to GNodeLib : use NPY arrays with glm::vec help
* GMergedMesh then takes it from there with the appropriate selections  
* where appropriate do things in static methods of other classes



High Level Geometry Information Flow
----------------------------------------

0. Geant4 volume tree
1. recursive traversal of Geant4 tree (eg by X4PhysicalVolume::convertNode) yields GGeo::m_root tree of GVolume(GNode)
2. GInstancer labels the tree of GVolume with ridx (repeat index) integers with zero being for the non-instanced remainder
3. GMergedMesh::Create for each slot collects volumes for each instance and the remainder volumes into separate MM.


GMergedMesh shapes
---------------------

::

    epsilon:GMergedMesh blyth$ np.py */bbox.npy
    a :                                                   0/bbox.npy :           (12230, 6) : 606b84624e6fb20a35a4050d3aef59be : 20200930-1120 
    b :                                                   6/bbox.npy :            (4486, 6) : 348eb6e0bdbc50a3184d5800bee497d4 : 20200930-1120 
    c :                                                   5/bbox.npy :               (5, 6) : 5929fc591e08d5308cb765783317002c : 20200930-1120 
    d :                                                   1/bbox.npy :               (1, 6) : 82523263e70e9ba4222142df304ecceb : 20200930-1120 
    e :                                                   2/bbox.npy :               (1, 6) : c0d0901849b5d5c0bd0673651fcfe526 : 20200930-1120 
    f :                                                   3/bbox.npy :               (1, 6) : eb467bed8841503e6664ccde21ee03cc : 20200930-1120 
    g :                                                   4/bbox.npy :               (1, 6) : 19dfce8e6901a007a2608b0826363b3b : 20200930-1120 
    epsilon:GMergedMesh blyth$ np.py */center_extent.npy
    a :                                          0/center_extent.npy :           (12230, 4) : 21957ef1c2a90ab18ed1729b02fa7aaa : 20200930-1120 
    b :                                          6/center_extent.npy :            (4486, 4) : 7b45bf3c1a48c091bcab9fb22958d369 : 20200930-1120 
    c :                                          5/center_extent.npy :               (5, 4) : 923d7b031cae87410b851a946cfa2e61 : 20200930-1120 
    d :                                          1/center_extent.npy :               (1, 4) : 43ebe68314a1c4d2f1485a8f17cd8e7d : 20200930-1120 
    e :                                          2/center_extent.npy :               (1, 4) : e3b4bc514a86d7c3d4a461e427edf72c : 20200930-1120 
    f :                                          3/center_extent.npy :               (1, 4) : 8e68cbc6208878db707f29841f2fad23 : 20200930-1120 
    g :                                          4/center_extent.npy :               (1, 4) : adec42edc7598e0656f913cf8edc0ad0 : 20200930-1120 
    epsilon:GMergedMesh blyth$ np.py */identity.npy
    a :                                               0/identity.npy :           (12230, 4) : dc2a1a0dd35dfa221e8bc891c52e1ec9 : 20200930-1120 
    b :                                               6/identity.npy :            (4486, 4) : e635d175b5626e3320b819b22653614f : 20200930-1120 
    c :                                               5/identity.npy :               (5, 4) : e42b6abf4d286c779e42758582e1a8dc : 20200930-1120 
    d :                                               1/identity.npy :               (1, 4) : 6f162e0cd93d44401363c8340a819f52 : 20200930-1120 
    e :                                               2/identity.npy :               (1, 4) : 77f1c534a138c9288e366029de2798fa : 20200930-1120 
    f :                                               3/identity.npy :               (1, 4) : 672223291a268328cd8890754dd29f7d : 20200930-1120 
    g :                                               4/identity.npy :               (1, 4) : c30fa39c1f6b03dc6aa0a12f67cba8bf : 20200930-1120 

    epsilon:GMergedMesh blyth$ np.py */nodeinfo.npy 
    a :                                               0/nodeinfo.npy :           (12230, 4) : ee5b2544536e9b5ee18d7fbffdd8807d : 20200930-1120 
    b :                                               6/nodeinfo.npy :            (4486, 4) : 4d749cd8c64bd24a1e79adfab2be9bf9 : 20200930-1120 
    c :                                               5/nodeinfo.npy :               (5, 4) : a2872b32a9b3e9384c7aa48474c772c6 : 20200930-1120 
    d :                                               1/nodeinfo.npy :               (1, 4) : 3cb60b0e0e0e39aa6d183f068b72e5a5 : 20200930-1120 
    e :                                               2/nodeinfo.npy :               (1, 4) : 791800b52346aaaada39469ed5bf5a84 : 20200930-1120 
    f :                                               3/nodeinfo.npy :               (1, 4) : 6d389f7e8991f94db981f94c8e74441f : 20200930-1120 
    g :                                               4/nodeinfo.npy :               (1, 4) : 2ac94e70eefa3f67d14e90a7ad1a0ebb : 20200930-1120 
    epsilon:GMergedMesh blyth$ np.py */meshes.npy 
    a :                                                 0/meshes.npy :           (12230, 1) : 3f9f703c8d1653785f7d40d9a77cddac : 20200930-1120 
    b :                                                 6/meshes.npy :            (4486, 1) : b93a589c54ffdf8c4b7bc8c2cca707e8 : 20200930-1120 
    c :                                                 5/meshes.npy :               (5, 1) : 23995356f32ef1ef90314c385c3a688d : 20200930-1120 
    d :                                                 1/meshes.npy :               (1, 1) : ad4d8518127a50b1bca320c052e3a369 : 20200930-1120 
    e :                                                 2/meshes.npy :               (1, 1) : 95de8a539bb8958fae8033f034876b8c : 20200930-1120 
    f :                                                 3/meshes.npy :               (1, 1) : a79e4a2fe7e25fdef237a41bacdcc8a4 : 20200930-1120 
    g :                                                 4/meshes.npy :               (1, 1) : 4439f62208a37f016af47a55767d2253 : 20200930-1120 







    epsilon:GMergedMesh blyth$ np.py */vertices.npy 
    a :                                               0/vertices.npy :          (247718, 3) : c22ae90461bbc0f34253fdb894b732d4 : 20200930-1120 
    b :                                               6/vertices.npy :          (247718, 3) : c22ae90461bbc0f34253fdb894b732d4 : 20200930-1120 
    c :                                               5/vertices.npy :            (1498, 3) : 5ee8d9f7a22054442dbadd9f00ef205c : 20200930-1120 
    d :                                               1/vertices.npy :               (8, 3) : 1bccb28b2613eb38fdfc5dc13688a5bd : 20200930-1120 
    e :                                               2/vertices.npy :               (8, 3) : e0075e455073dc682ef02160c655b3cb : 20200930-1120 
    f :                                               3/vertices.npy :               (8, 3) : d78516c266c051959587fcf4fd18b387 : 20200930-1120 
    g :                                               4/vertices.npy :               (8, 3) : 6df15698bc7a298f8bcdbb9ab28eba1a : 20200930-1120 

    epsilon:GMergedMesh blyth$ np.py */boundaries.npy 
    a :                                             0/boundaries.npy :          (480972, 1) : ff2d347e3c3de52e03c31ace0ba4e833 : 20200930-1120 
    b :                                             6/boundaries.npy :          (480972, 1) : ff2d347e3c3de52e03c31ace0ba4e833 : 20200930-1120 
    c :                                             5/boundaries.npy :            (2976, 1) : c092ab645e1e555693e2267fcc552395 : 20200930-1120 
    d :                                             1/boundaries.npy :              (12, 1) : dcb4346e43ee94d14fe59f6d5735607e : 20200930-1120 
    e :                                             2/boundaries.npy :              (12, 1) : f76afd417acf546cc61af59aa09c94fa : 20200930-1120 
    f :                                             3/boundaries.npy :              (12, 1) : f7d71121ab65a8b662d8fb366e9b866f : 20200930-1120 
    g :                                             4/boundaries.npy :              (12, 1) : b1717c5104028d47368bb72c600d0050 : 20200930-1120 




    epsilon:GMergedMesh blyth$ np.py */iidentity.npy
    a :                                              1/iidentity.npy :         (1792, 1, 4) : 54ccef21c5e74ec53cd6f1ea49112044 : 20200930-1120 
    b :                                              2/iidentity.npy :          (864, 1, 4) : d9a4c0bbe91c9a2cba8fdc08397d26eb : 20200930-1120 
    c :                                              3/iidentity.npy :          (864, 1, 4) : d40cd53bb48e8505da25237766000e90 : 20200930-1120 
    d :                                              4/iidentity.npy :          (864, 1, 4) : 1b7fb9d7357be6d29363e97d4d265d6f : 20200930-1120 
    e :                                              5/iidentity.npy :          (672, 5, 4) : 3bc94f5be5b366b94658ed846214f37d : 20200930-1120 
    f :                                              0/iidentity.npy :         (1, 4486, 4) : a4562b3dca31821d7565956d4a7f4d2c : 20200930-1120 
    g :                                              6/iidentity.npy :         (1, 4486, 4) : a4562b3dca31821d7565956d4a7f4d2c : 20200930-1120 
    epsilon:GMergedMesh blyth$ np.py */itransforms.npy
    a :                                            1/itransforms.npy :         (1792, 4, 4) : 629c8b792e4965ab2080904c53625398 : 20200930-1120 
    b :                                            2/itransforms.npy :          (864, 4, 4) : cb1febd543aec99c5a56158e5c0b83f5 : 20200930-1120 
    c :                                            3/itransforms.npy :          (864, 4, 4) : d8ea1072b35e4bdcc8e2375920da4b53 : 20200930-1120 
    d :                                            4/itransforms.npy :          (864, 4, 4) : 3d0a86f012d6b331105d27aa7914cd2e : 20200930-1120 
    e :                                            5/itransforms.npy :          (672, 4, 4) : 684f8b4688efd18ffab00c1910ad5dc7 : 20200930-1120 
    f :                                            0/itransforms.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200930-1120 
    g :                                            6/itransforms.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200930-1120 

    ## hmm transforms within the instance not here (all identity in DYB and JUNO) 

    epsilon:GMergedMesh blyth$ echo $(( 1792+864+864+864+672*5+4486 ))
    12230



::

    epsilon:1 blyth$ np.py 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/GMergedMesh/1

    . :                                             ./transforms.npy :              (1, 16) : 741176dbe55e7a88023d21fa0bc838d7 : 20200930-1120 
    . :                                                   ./bbox.npy :               (1, 6) : 82523263e70e9ba4222142df304ecceb : 20200930-1120 
    . :                                          ./center_extent.npy :               (1, 4) : 43ebe68314a1c4d2f1485a8f17cd8e7d : 20200930-1120 
    . :                                               ./identity.npy :               (1, 4) : 6f162e0cd93d44401363c8340a819f52 : 20200930-1120 
    . :                                               ./nodeinfo.npy :               (1, 4) : 3cb60b0e0e0e39aa6d183f068b72e5a5 : 20200930-1120 
    . :                                                 ./meshes.npy :               (1, 1) : ad4d8518127a50b1bca320c052e3a369 : 20200930-1120 

    . :                                                 ./colors.npy :               (8, 3) : ccce7249abc8b71fafe2504b83d3adff : 20200930-1120 
    . :                                                ./normals.npy :               (8, 3) : dde5918e0975159819e6ad30ebce37ef : 20200930-1120 
    . :                                               ./vertices.npy :               (8, 3) : 1bccb28b2613eb38fdfc5dc13688a5bd : 20200930-1120 
    8 vtx  

    . :                                             ./boundaries.npy :              (12, 1) : dcb4346e43ee94d14fe59f6d5735607e : 20200930-1120 
    . :                                                  ./nodes.npy :              (12, 1) : dcb4346e43ee94d14fe59f6d5735607e : 20200930-1120 
    . :                                                ./sensors.npy :              (12, 1) : d271e4911977444efba376cd91a1bfdc : 20200930-1120 
    . :                                                ./indices.npy :              (36, 1) : 1c3806f5183e168f7f820fa91fd1d88f : 20200930-1120 
                                  12*3 = 36  TODO:  (36,1) -> (12,3)  
    12 tri : from triangulated cube  


    . :                                              ./iidentity.npy :         (1792, 1, 4) : 54ccef21c5e74ec53cd6f1ea49112044 : 20200930-1120 
    . :                                            ./itransforms.npy :         (1792, 4, 4) : 629c8b792e4965ab2080904c53625398 : 20200930-1120 
    1792 placements


    epsilon:5 blyth$ np.py *.npy
    (face level)
    a :                                                  indices.npy :            (8928, 1) : ea75c0fb642b2ffc6b2a5d3410af2f77 : 20200930-1120 
    b :                                               boundaries.npy :            (2976, 1) : c092ab645e1e555693e2267fcc552395 : 20200930-1120 
    c :                                                    nodes.npy :            (2976, 1) : 615f3a63b87205fd675b15c572fd6737 : 20200930-1120 
    d :                                                  sensors.npy :            (2976, 1) : 8973840b863d4b6d1250a77979216631 : 20200930-1120 

    (vertex level)
    e :                                                   colors.npy :            (1498, 3) : e0568a419833e257bfe1712b8565a94d : 20200930-1120 
    f :                                                  normals.npy :            (1498, 3) : 0b2bd932335556ec5750e42d650a6728 : 20200930-1120 
    g :                                                 vertices.npy :            (1498, 3) : 5ee8d9f7a22054442dbadd9f00ef205c : 20200930-1120 

    (volume level)
    j :                                                     bbox.npy :               (5, 6) : 5929fc591e08d5308cb765783317002c : 20200930-1120 
    k :                                            center_extent.npy :               (5, 4) : 923d7b031cae87410b851a946cfa2e61 : 20200930-1120 
    l :                                                 identity.npy :               (5, 4) : e42b6abf4d286c779e42758582e1a8dc : 20200930-1120 
    m :                                                   meshes.npy :               (5, 1) : 23995356f32ef1ef90314c385c3a688d : 20200930-1120 
    n :                                                 nodeinfo.npy :               (5, 4) : a2872b32a9b3e9384c7aa48474c772c6 : 20200930-1120 
    o :                                               transforms.npy :              (5, 16) : 90bdb3bf884fcaf38a71d524190e2304 : 20200930-1120 

    (placement level)
    h :                                                iidentity.npy :          (672, 5, 4) : 3bc94f5be5b366b94658ed846214f37d : 20200930-1120 
    i :                                              itransforms.npy :          (672, 4, 4) : 684f8b4688efd18ffab00c1910ad5dc7 : 20200930-1120 


    To clarify these groupings have prefixed the names.




    epsilon:6 blyth$ np.py *.npy  "globalinstance"
    a :                                                  indices.npy :         (1442916, 1) : 77c79d95ccf148e00ac5057d5c5312e3 : 20200930-1120 
    b :                                               boundaries.npy :          (480972, 1) : ff2d347e3c3de52e03c31ace0ba4e833 : 20200930-1120 
    c :                                                    nodes.npy :          (480972, 1) : c1ac1e3bd7affa2fdccd215c6acb04f1 : 20200930-1120 
    d :                                                  sensors.npy :          (480972, 1) : 25e46a82c3bf1da3dd23fc9f4f38179a : 20200930-1120 

    e :                                                   colors.npy :          (247718, 3) : 20ff305b06166e347fac1c642f963578 : 20200930-1120 
    f :                                                  normals.npy :          (247718, 3) : c587f16c54aa1aa9cb9f94d526b03210 : 20200930-1120 
    g :                                                 vertices.npy :          (247718, 3) : c22ae90461bbc0f34253fdb894b732d4 : 20200930-1120 

    h :                                                     bbox.npy :            (4486, 6) : 348eb6e0bdbc50a3184d5800bee497d4 : 20200930-1120 
    i :                                            center_extent.npy :            (4486, 4) : 7b45bf3c1a48c091bcab9fb22958d369 : 20200930-1120 
    j :                                                 identity.npy :            (4486, 4) : e635d175b5626e3320b819b22653614f : 20200930-1120 
    k :                                                   meshes.npy :            (4486, 1) : b93a589c54ffdf8c4b7bc8c2cca707e8 : 20200930-1120 
    l :                                                 nodeinfo.npy :            (4486, 4) : 4d749cd8c64bd24a1e79adfab2be9bf9 : 20200930-1120 
    m :                                               transforms.npy :           (4486, 16) : 85360b6de1a60e8246272019869cba09 : 20200930-1120 

    n :                                                iidentity.npy :         (1, 4486, 4) : a4562b3dca31821d7565956d4a7f4d2c : 20200930-1120 
    o :                                              itransforms.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200930-1120 
    epsilon:6 blyth$ 


    epsilon:0 blyth$ np.py *.npy   unselected 
    a :                                                  indices.npy :         (1442916, 1) : 77c79d95ccf148e00ac5057d5c5312e3 : 20200930-1120 
    b :                                               boundaries.npy :          (480972, 1) : ff2d347e3c3de52e03c31ace0ba4e833 : 20200930-1120 
    c :                                                    nodes.npy :          (480972, 1) : c1ac1e3bd7affa2fdccd215c6acb04f1 : 20200930-1120 
    d :                                                  sensors.npy :          (480972, 1) : 25e46a82c3bf1da3dd23fc9f4f38179a : 20200930-1120 

    e :                                                   colors.npy :          (247718, 3) : 879d0c4dad015355d5af3e2d14dee5b7 : 20200930-1120 
    f :                                                  normals.npy :          (247718, 3) : c587f16c54aa1aa9cb9f94d526b03210 : 20200930-1120 
    g :                                                 vertices.npy :          (247718, 3) : c22ae90461bbc0f34253fdb894b732d4 : 20200930-1120 

    h :                                                     bbox.npy :           (12230, 6) : 606b84624e6fb20a35a4050d3aef59be : 20200930-1120 
    i :                                            center_extent.npy :           (12230, 4) : 21957ef1c2a90ab18ed1729b02fa7aaa : 20200930-1120 
    j :                                                 identity.npy :           (12230, 4) : dc2a1a0dd35dfa221e8bc891c52e1ec9 : 20200930-1120 
    k :                                                   meshes.npy :           (12230, 1) : 3f9f703c8d1653785f7d40d9a77cddac : 20200930-1120 
    l :                                                 nodeinfo.npy :           (12230, 4) : ee5b2544536e9b5ee18d7fbffdd8807d : 20200930-1120 
    m :                                               transforms.npy :          (12230, 16) : 6e74cf2cd82feb99da06b58f069c8985 : 20200930-1120 
    ## all volume here is just confusing 

    n :                                                iidentity.npy :         (1, 4486, 4) : a4562b3dca31821d7565956d4a7f4d2c : 20200930-1120 
    o :                                              itransforms.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200930-1120 
    epsilon:0 blyth$ 




Can meshes be removed ?  SEEMS YES : BUT NEED TO FIND USAGE
-------------------------------------------------------------------

::

    epsilon:0 blyth$ ipython

    In [1]: m = np.load("meshes.npy")                                                                                                                                                                    

    In [2]: m                                                                                                                                                                                            
    Out[2]: 
    array([[248],
           [247],
           [ 21],
           ...,
           [243],
           [244],
           [245]], dtype=uint32)

    In [3]: i = np.load("identity.npy")                                                                                                                                                                  

    In [4]: i                                                                                                                                                                                            
    Out[4]: 
    array([[    0,   248,     0,     0],
           [    1,   247,     1,     0],
           [    2,    21,     2,     0],
           ...,
           [12227,   243,   126,     0],
           [12228,   244,   126,     0],
           [12229,   245,   126,     0]], dtype=uint32)





