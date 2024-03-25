SScene_test_reveals_SMesh_polygonization_issue_with_PMT_virtual_wrapper_shape
================================================================================

FIXED : bug in grabbing the polygonization ? was skipping too much. 

::

    ~/o/sysrap/tests/SScene_test.sh 
    ...
    SScene::initMesh [107] skip YES so HamamatsuR12860_PMT_20inch_pmt_solid_1_40xa0a6940
    SScene::initMesh [108] skip NO  so HamamatsuR12860sMask_virtual0xa0b8450
    SMesh::SmoothNormals FATAL NOT expected  i [168]  t [ivec3(72, 99, 73)] num_vtx 98 num_tri 192
    Assertion failed: (y_expected), function SmoothNormals, file ../SMesh.h, line 462.
    /Users/blyth/o/sysrap/tests/SScene_test.sh: line 61: 16176 Abort trap: 6           $bin
    /Users/blyth/o/sysrap/tests/SScene_test.sh run error
    epsilon:opticks blyth$ 


With below in U4Mesh_test.sh::

     81 if [ "${arg/view}" != "$arg" ]; then
     82 
     83     export FOLD=/data/blyth/opticks/U4TreeCreateTest/stree/mesh
     84     export SOLID=HamamatsuR12860sMask_virtual0xa0b8450
     85 
     86     ${IPYTHON:-ipython} --pdb -i $name.py
     87     [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
     88 fi

     ~/o/u4/tests/U4Mesh_test.sh view

The view shows expected external cylinder and cut cone, but 
looking from below shows unexpected inner of single cutaway 
cone.::

    epsilon:~ blyth$ cd /data/blyth/opticks/U4TreeCreateTest/stree/mesh/HamamatsuR12860sMask_virtual0xa0b8450
    epsilon:HamamatsuR12860sMask_virtual0xa0b8450 blyth$ f
    f

    CMDLINE:/Users/blyth/np/f.py
    f.base:.

      : f.vtx                                              :              (98, 3) : 17:54:26.814050 
      : f.tpd                                              :               (768,) : 17:54:26.813028 
      : f.tri                                              :             (192, 3) : 17:54:26.813244 
      : f.NPFold_index                                     :                 (5,) : 17:54:26.812769 
      : f.fpd                                              :               (552,) : 17:54:26.813825 
      : f.NPFold_names                                     :                 (0,) : 17:54:26.812606 
      : f.face                                             :             (120, 4) : 17:54:26.813450 

     min_stamp : 2024-03-16 17:27:05.460837 
     max_stamp : 2024-03-16 17:27:05.462281 
     dif_stamp : 0:00:00.001444 
     age_stamp : 17:54:26.812606 



f.tri shows lots of ref to non-existing vertex 99::

       [96, 22, 23],
       [96, 23,  0],
       [72, 99, 73],
       [73, 99, 74],
       [74, 99, 75],
       [75, 99, 76],
       [76, 99, 77],
       [77, 99, 78],
       [78, 99, 79],
       [79, 99, 80],
       [80, 99, 81],
       [81, 99, 82],
       [82, 99, 83],
       [83, 99, 84],
       [84, 99, 85],
       [85, 99, 86],
       [86, 99, 87],
       [87, 99, 88],
       [88, 99, 89],


TODO: U4Mesh dumping of poly collection, and check::

    148 inline void U4Mesh::init_vtx_face_count()
    149 {
    150     for(int i=0 ; i < nf ; i++)
    151     {
    152         G4int nedge;
    153         G4int ivertex[4];
    154         G4int iedgeflag[4];
    155         G4int ifacet[4];
    156         poly->GetFacet(i+1, nedge, ivertex, iedgeflag, ifacet);
    157         assert( nedge == 3 || nedge == 4  );
    158 
    159         for(int j=0 ; j < nedge ; j++)
    160         {
    161             G4int iv = ivertex[j] - 1 ;
    162             if(v2fc.count(iv) == 0)
    163             {
    164                v2fc[iv] = 1 ;
    165             }
    166             else
    167             {
    168                 v2fc[iv] += 1 ;
    169             }
    170         }
    171     }
    172 
    173     nv = v2fc.size();
    174     //std::cout << desc_vtx_face_count() ; 
    175 }



::

   ~/o/u4/tests/U4TreeCreateTest.sh  


After switchin off the U4Tree disqualify the view looks better.



