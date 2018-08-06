material_matching_between_old_and_direct_routes
==================================================

This spawned from 

* :doc:`direct_route_boundary_match`
* :doc:`surface_ordering`


Context : Workflow comparing old Assimp/G4DAE route with X4 direct route
--------------------------------------------------------------------------

1. run the old route::

   op --gdml2gltf
   OPTICKS_RESOURCE_LAYOUT=104 OKTest -G --gltf 3  

   ## loads G4DAE assimp + the analytic description from python


2. run the new route, using the geocache written by the above for fixup::

   OPTICKS_RESOURCE_LAYOUT=104 OKX4Test  

   ## loads GDML to conjure up the G4 model, fixes omissions with G4DAE,
   ## then converts the G4 model directly to Opticks GGeo   

3. NumPy geometry comparisons:

   * ab-l
   * ab-mat
   * ab-mat1


Controlling material order prior to X4 (direct code)
------------------------------------------------------

Want the X4 code to just follow the Geant4 order.

* added sortMaterials fixup using CMaterialSort to CGDMLDetector::init 
  loading a sorting of the material table at G4 level to follow the preference order.
 
  This is before getting to X4 level so X4 direct code just follows the G4 ordering it sees.  


This gets close to an order match : apart from GlassSchottF2 which is not present
in the GDML, its added later at Opticks level.  So change the order opticksdata/export/DayaBay/GMaterialLib/order.json
to prevent the test materials partaking::


     16     "Air": "15",
     17     "GlassSchottF2": "160",
     18     "PPE": "17",

After that the ordering matches. ab-mat ::

    epsilon:opticks blyth$ hg commit -m "material order now matching, using pref order sort for old route and CGDMLDetector::sortMaterials fixup prior to direct route "


Material Ordering 
--------------------

* A is ordered into the preference order : actually there is good reason to 
  keep that order ... 

* A with sorting switching off (GMaterialLib) get something close
  to alphabetical : Probably ColladaLoader map mangling again. FIXED via srcidx property 

* B X4MaterialTable::init is just following the G4 order, getting same order 
  as in the GDML 


Keeping source order thru COLLADA stage
-------------------------------------------

::

    g4-cls G4MaterialTable   ## just a vector of G4Material


Added a g4dae_material_srcidx property to ColladaParser and 
grabbed that in AssimpGGeo and set it on GPropertyMap MetaKV of
the materials.  This allowing to sort the G4DAE loaded materials
by their original Geant4 ordering. 

Modulo the test materials (which should be stuffed on then end and not sorted, 
or set to an appropriate srcidx to make them stay in their place)
this gets the same ordering for A and B : the original G4 ordering.  

But its not the desired ordering ...


Material content ab-mat1
---------------------------

Huh all groupvel stuck at 300 in A, but B is varying as expect::

    In [42]: np.all( a[:,1,:,0] == 300. )
    Out[42]: True



SMOKING BUG with GROUPVEL : ITS ALWAYS THE SAME
------------------------------------------------


Dumping pointers, all the vg are the same

::

     i  0 n 38 mat 0x7fcbba0ebad0 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba432e30 k __dd__Materials__ADTableStainlessSteel0xc177178
     i  1 n 38 mat 0x7fcbba0ed730 vg 0x7fcbb5767440 ri 0x7fcbba0ed6b0 vg_calc 0x7fcbba438820 k __dd__Materials__Acrylic0xc02ab98
     i  2 n 38 mat 0x7fcbba0ef310 vg 0x7fcbb5767440 ri 0x7fcbba0ef280 vg_calc 0x7fcbba439300 k __dd__Materials__Air0xc032550
     i  3 n 38 mat 0x7fcbba0f05a0 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba439de0 k __dd__Materials__Aluminium0xc542070
     i  4 n 38 mat 0x7fcbba0f11e0 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba43a8c0 k __dd__Materials__BPE0xc0ad360
     i  5 n 38 mat 0x7fcbba0f1ea0 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba43b3a0 k __dd__Materials__Bakelite0xc2bc240
     i  6 n 38 mat 0x7fcbba0f3690 vg 0x7fcbb5767440 ri 0x7fcbba0f35a0 vg_calc 0x7fcbba43be80 k __dd__Materials__Bialkali0xc2f2428
     i  7 n 38 mat 0x7fcbba0f4de0 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba43c960 k __dd__Materials__C_130xc3d0ab0
     i  8 n 38 mat 0x7fcbba0f59d0 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba43d440 k __dd__Materials__Co_600xc3cf0c0
     i  9 n 38 mat 0x7fcbba0f6b30 vg 0x7fcbb5767440 ri 0x7fcbba0f6e10 vg_calc 0x7fcbba43df20 k __dd__Materials__DeadWater0xbf8a548
     i 10 n 38 mat 0x7fcbba0f8670 vg 0x7fcbb5767440 ri 0x7fcbb5764990 vg_calc 0x7fcbba43ea00 k __dd__Materials__ESR0xbf9f438





::

    In [22]: a[30:,:,:18,0]
    Out[22]: 
    array([[[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.6807,   1.669 ,   1.6598,   1.6523],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.3776,   1.3664,   1.3588,   1.353 ,   1.349 ,   1.3466,   1.3442,   1.3422,   1.3406,   1.339 ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]]],
          dtype=float32)

    In [23]: b[30:,:,:18,0]
    Out[23]: 
    array([[[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924]],

           [[  1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ,   1.    ],
            [299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924, 299.7924]],

           [[  1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.696 ,   1.6807,   1.669 ,   1.6598,   1.6523],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]],

           [[  1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.396 ,   1.3776,   1.3664,   1.3588,   1.353 ,   1.349 ,   1.3466,   1.3442,   1.3422,   1.3406,   1.339 ],
            [214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 214.751 , 201.0058, 195.2175, 202.9706, 207.4884, 211.0383, 214.7387, 216.4586, 217.0239, 218.1289, 218.668 , 220.0235]]],
          dtype=float32)

    In [24]: 



Attempt to move replaceGROUPVEL to beforeClose giving different result ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


B invokes GMaterialLib::replaceGROUPVEL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    211 void X4PhysicalVolume::convertMaterials()
    212 {
    213     LOG(fatal) << "[" ;
    214 
    215     size_t num_materials0 = m_mlib->getNumMaterials() ;
    216     assert( num_materials0 == 0 );
    217 
    218     X4MaterialTable::Convert(m_mlib);
    219 
    220     size_t num_materials = m_mlib->getNumMaterials() ;
    221     assert( num_materials > 0 );
    222 
    223 
    224     LOG(fatal) << "."
    225                << " num_materials " << num_materials
    226                ;
    227 
    228 
    229     // TODO : can these go into one method within GMaterialLib?
    230     m_mlib->addTestMaterials() ;
    231 
    232     m_mlib->close();   // may change order if prefs dictate
    233 
    234     // replaceGROUPVE needs the buffer : so must be after close
    235     bool debug = false ;
    236     m_mlib->replaceGROUPVEL(debug);
    237 


::

    epsilon:extg4 blyth$ opticks-find replaceGROUPVEL
    ./extg4/X4PhysicalVolume.cc:    m_mlib->replaceGROUPVEL(debug); 
    ./ggeo/GMaterialLib.cc:       replaceGROUPVEL(debug);
    ./ggeo/GMaterialLib.cc:void GMaterialLib::replaceGROUPVEL(bool debug)
    ./ggeo/GMaterialLib.cc:    LOG(info) << "GMaterialLib::replaceGROUPVEL " << " ni " << ni ;
    ./ggeo/GMaterialLib.cc:    const char* base = "$TMP/replaceGROUPVEL" ;
    ./ggeo/GMaterialLib.cc:        LOG(warning) << "GMaterialLib::replaceGROUPVEL debug active : saving refractive_index.npy and group_velocity.npy beneath " << base  ; 
    ./ggeo/GMaterialLib.cc:            dump(mat, "replaceGROUPVEL");
    ./ggeo/GMaterialLib.hh:       void replaceGROUPVEL(bool debug=false);  // triggered in postLoadFromCache with --groupvel option
    ./ana/groupvel.py:class replaceGROUPVEL(PropTree):
    ./ana/groupvel.py:    GMaterialLib::replaceGROUPVEL in debug mode writes the 
    ./ana/groupvel.py:    base = "$TMP/replaceGROUPVEL"
    ./ana/groupvel.py:    rg = replaceGROUPVEL()
    epsilon:opticks blyth$ 


The reason is that the replaceGROUPVEL is done GMaterialLib::postLoadFromCache in the old route, so as are comparing 
the raw geocache miss out on the change.

* can this be moved precache ?

::

     057 void GMaterialLib::postLoadFromCache()
      58 {
      ..
      70 
      71     bool groupvel = !m_ok->hasOpt("nogroupvel") ;
      72 
     121     if(groupvel)  // unlike the other material changes : this one is ON by default, so long at not swiched off with --nogroupvel 
     122     {
     123        bool debug = false ;
     124        replaceGROUPVEL(debug);
     125     }
     126 
     127     if(nore || noab || nosc || xxre || xxab || xxsc || fxre || fxsc || fxab || groupvel)
     128     {
     129         // need to replace the loaded buffer with a new one with the changes for Opticks to see it 
     130         NPY<float>* mbuf = createBuffer();
     131         setBuffer(mbuf);
     132     }
     133 
     134 }




