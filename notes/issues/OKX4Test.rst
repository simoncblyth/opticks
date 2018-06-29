okg4.OKX4Test : checking direct from G4 conversion, starting from a GDML loaded geometry
===========================================================================================

Investigate some missing geometry in the visualization
----------------------------------------------------------

Run without args::

    OKX4Test


Observations from visualization
----------------------------------

* missing some geometry at edge of pool, maybe girders are the X4Mesh skipped lvIdx ? 

  * bizarrely these showed up without clear cause after GMeshLib code cleanup 

* Coloring very different, maybe not picking up dyb prefs for "g4live" detector 
* geometry normals visible
* triangulated ray trace is notably faster that usual (missing some expensive geometry ?)    


Observations from GGeo::save
--------------------------------



partBuffer differences
~~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`OKX4Test_partBuffer_difference`



GParts::save mm5 one less transform ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [1]: a = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/primBuffer.npy")

    In [2]: a
    Out[2]: 
    array([[ 0, 15,  0,  0],
           [15, 15,  4,  0],
           [30,  7,  8,  0],
           [37,  3, 10,  0],
           [40,  1, 11,  0]], dtype=int32)

    In [3]: b = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/primBuffer.npy")

    In [4]: b
    Out[4]: 
    array([[ 0, 15,  0,  0],
           [15, 15,  4,  0],
           [30,  7,  8,  0],
           [37,  3,  9,  0],
           [40,  1, 10,  0]], dtype=int32)

                 ^
               num-parts always complete binary tree size 

    In [5]: 

::

    In [5]: a = np.load("/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5/tranBuffer.npy")

    In [6]: b = np.load("/usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts/5/tranBuffer.npy")


    In [17]: a[:,0,3,:]   no-rotation so just look at last row 
    Out[17]: 
    array([[  0. ,   0. ,   0. ,   1. ],
           [  0. ,   0. ,  43. ,   1. ],
           [  0. ,   0. ,  69. ,   1. ],
           [  0. ,   0. , -84.5,   1. ],
           [  0. ,   0. ,   0. ,   1. ],
           [  0. ,   0. ,  43. ,   1. ],
           [  0. ,   0. ,  69. ,   1. ],
           [  0. ,   0. , -81.5,   1. ],
           [  0. ,   0. ,   0. ,   1. ],
           [  0. ,   0. ,  43. ,   1. ],  <<<<  missing 
           [  0. ,   0. ,  69. ,   1. ],
           [  0. ,   0. , -81.5,   1. ]], dtype=float32)

    In [18]: b[:,0,3,:]
    Out[18]: 
    array([[  0. ,   0. ,   0. ,   1. ],
           [  0. ,   0. ,  43. ,   1. ],
           [  0. ,   0. ,  69. ,   1. ],
           [  0. ,   0. , -84.5,   1. ],
           [  0. ,   0. ,   0. ,   1. ],
           [  0. ,   0. ,  43. ,   1. ],
           [  0. ,   0. ,  69. ,   1. ],
           [  0. ,   0. , -81.5,   1. ],
           [  0. ,   0. ,   0. ,   1. ],
           [  0. ,   0. ,  69. ,   1. ],
           [  0. ,   0. , -81.5,   1. ]], dtype=float32)

    In [19]: 





GParts::makePrimBuffer::

    1008             pri.x = part_offset ;
    1009             pri.y = m_primflag == CSG_FLAGPARTLIST ? -parts_for_prim : parts_for_prim ;
    1010             pri.z = tran_offset ;
    1011             pri.w = plan_offset ;
    1012 




FIXED : GParts::save missing tran+plan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only part and prim show up, no tran or plan ?

* fixed by adding::

   NCSG::export_node
   NCSG::export_gtransform 
   NCSG::export_planes



After fix::

    epsilon:GParts blyth$ np.py 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts
          ./0/GParts.txt : 85264               #### vs 11984  :  why so many more ???
      ./0/planBuffer.npy : (672, 4) 
      ./0/partBuffer.npy : (85264, 4, 4) 
      ./0/tranBuffer.npy : (5256, 3, 4, 4)     ##### vs 5344
      ./0/primBuffer.npy : (3116, 4) 
          ./4/GParts.txt : 1 
      ./4/partBuffer.npy : (1, 4, 4) 
      ./4/tranBuffer.npy : (1, 3, 4, 4) 
      ./4/primBuffer.npy : (1, 4) 
          ./3/GParts.txt : 1 
      ./3/partBuffer.npy : (1, 4, 4) 
      ./3/tranBuffer.npy : (1, 3, 4, 4) 
      ./3/primBuffer.npy : (1, 4) 
          ./2/GParts.txt : 1 
      ./2/partBuffer.npy : (1, 4, 4) 
      ./2/tranBuffer.npy : (1, 3, 4, 4) 
      ./2/primBuffer.npy : (1, 4) 
          ./5/GParts.txt : 41 
      ./5/partBuffer.npy : (41, 4, 4) 
      ./5/tranBuffer.npy : (11, 3, 4, 4)      ##### huh one less transform ???  Goes to 12 when dont restrict to primitives 
      ./5/primBuffer.npy : (5, 4) 
    epsilon:GParts blyth$ 


Compared with source cache::

    epsilon:GPartsAnalytic blyth$ np.py 
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic
          ./0/GParts.txt : 11984 
      ./0/planBuffer.npy : (672, 4) 
      ./0/partBuffer.npy : (11984, 4, 4) 
      ./0/tranBuffer.npy : (5344, 3, 4, 4) 
      ./0/primBuffer.npy : (3116, 4) 
          ./4/GParts.txt : 1 
      ./4/partBuffer.npy : (1, 4, 4) 
      ./4/tranBuffer.npy : (1, 3, 4, 4) 
      ./4/primBuffer.npy : (1, 4) 
          ./3/GParts.txt : 1 
      ./3/partBuffer.npy : (1, 4, 4) 
      ./3/tranBuffer.npy : (1, 3, 4, 4) 
      ./3/primBuffer.npy : (1, 4) 
          ./2/GParts.txt : 1 
      ./2/partBuffer.npy : (1, 4, 4) 
      ./2/tranBuffer.npy : (1, 3, 4, 4) 
      ./2/primBuffer.npy : (1, 4) 
          ./5/GParts.txt : 41 
      ./5/partBuffer.npy : (41, 4, 4) 
      ./5/tranBuffer.npy : (12, 3, 4, 4) 
      ./5/primBuffer.npy : (5, 4) 
    epsilon:GPartsAnalytic blyth$ 





Before fix::

    epsilon:GParts blyth$ np.py 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GParts
          ./0/GParts.txt : 85264 
      ./0/partBuffer.npy : (85264, 4, 4) 
      ./0/primBuffer.npy : (3116, 4) 
          ./4/GParts.txt : 1 
      ./4/partBuffer.npy : (1, 4, 4) 
      ./4/primBuffer.npy : (1, 4) 
          ./3/GParts.txt : 1 
      ./3/partBuffer.npy : (1, 4, 4) 
      ./3/primBuffer.npy : (1, 4) 
          ./2/GParts.txt : 1 
      ./2/partBuffer.npy : (1, 4, 4) 
      ./2/primBuffer.npy : (1, 4) 
          ./5/GParts.txt : 41 
      ./5/partBuffer.npy : (41, 4, 4) 
      ./5/primBuffer.npy : (5, 4) 
    epsilon:GParts blyth$ 


GParts::save::

     454     for(unsigned i=0 ; i < tags.size() ; i++)
     455     {
     456         const char* tag = tags[i].c_str();
     457         const char* name = BufferName(tag);
     458         NPY<float>* buf = getBuffer(tag);
     459         if(buf)
     460         {
     461             unsigned num_items = buf->getShape(0);
     462             if(num_items > 0)
     463             {
     464                 buf->save(dir, name);
     465             }
     466         }
     467     }
     468     if(m_prim_buffer) m_prim_buffer->save(dir, BufferName("prim"));


Tracing where the transforms come from
-----------------------------------------

::

    1209 unsigned NCSG::addUniqueTransform( const nmat4triple* gtransform_ )
    1210 {
    1211     bool no_offset = m_gpuoffset.x == 0.f && m_gpuoffset.y == 0.f && m_gpuoffset.z == 0.f ;
    1212 
    1213     bool reverse = true ; // <-- apply transfrom at root of transform hierarchy (rather than leaf)
    1214 
    1215     const nmat4triple* gtransform = no_offset ? gtransform_ : gtransform_->make_translated(m_gpuoffset, reverse, "NCSG::addUniqueTransform" ) ;
    1216 
    1217 
    1218     /*
    1219     std::cout << "NCSG::addUniqueTransform"
    1220               << " orig " << *gtransform_
    1221               << " tlated " << *gtransform
    1222               << " gpuoffset " << m_gpuoffset 
    1223               << std::endl 
    1224               ;
    1225     */
    1226 
    1227     NPY<float>* gtmp = NPY<float>::make(1,NTRAN,4,4);
    1228     gtmp->zero();
    1229     gtmp->setMat4Triple( gtransform, 0);
    1230 
    1231     unsigned gtransform_idx = 1 + m_gtransforms->addItemUnique( gtmp, 0 ) ;
    1232     delete gtmp ;
    1233     return gtransform_idx ;
    1234 }

Tis done on import::

    0970 nnode* NCSG::import_r(unsigned idx, nnode* parent)
     971 {
     972     if(idx >= m_num_nodes) return NULL ;
     973 
     974     OpticksCSG_t typecode = (OpticksCSG_t)getTypeCode(idx);
     975     int transform_idx = getTransformIndex(idx) ;
     976     bool complement = isComplement(idx) ;
     977 
     978     LOG(debug) << "NCSG::import_r"
     979               << " idx " << idx
     980               << " transform_idx " << transform_idx
     981               << " complement " << complement
     982               ;
     983 
     984     nnode* node = NULL ;  
     985 
     986     if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
     987     {   
     988         node = import_operator( idx, typecode ) ;
     989         
     990         node->parent = parent ;
     991         node->idx = idx ;  
     992         node->complement = complement ;
     993         
     994         node->transform = import_transform_triple( transform_idx ) ;
     995         
     996         node->left = import_r(idx*2+1, node ); 
     997         node->right = import_r(idx*2+2, node );
     998         
     999         node->left->other = node->right ;   // used by NOpenMesh 
    1000         node->right->other = node->left ;
    1001         
    1002         // recursive calls after "visit" as full ancestry needed for transform collection once reach primitives
    1003     }
    1004     else
    1005     {
    1006         node = import_primitive( idx, typecode );
    1007 
    1008         node->parent = parent ;                // <-- parent hookup needed prior to gtransform collection 
    1009         node->idx = idx ;
    1010         node->complement = complement ;
    1011 
    1012         node->transform = import_transform_triple( transform_idx ) ;
    1013 
    1014         const nmat4triple* gtransform = node->global_transform();
    1015 
    1016         // see opticks/notes/issues/subtree_instances_missing_transform.rst
    1017         //if(gtransform == NULL && m_usedglobally)
    1018         if(gtransform == NULL )  // move to giving all primitives a gtransform 
    1019         {
    1020             gtransform = nmat4triple::make_identity() ;
    1021         }




* hmm stuff done on import, never done in direct case 




How difficult to get rid of the m_analytic switch, and GScene ?
-------------------------------------------------------------------

* GGeo has m_analytic which is always false, and is passed along::

GGeo::init::

     358    //////////////  below only when operating pre-cache //////////////////////////
     359 
     360    m_bndlib = new GBndLib(m_ok);
     361    m_materiallib = new GMaterialLib(m_ok);
     362    m_surfacelib  = new GSurfaceLib(m_ok);
     363 
     364    m_bndlib->setMaterialLib(m_materiallib);
     365    m_bndlib->setSurfaceLib(m_surfacelib);
     366 
     367    // NB this m_analytic is always false
     368    //    the analytic versions of these libs are born in GScene
     369    assert( m_analytic == false );
     370    bool testgeo = false ;
     371 
     372    m_meshlib = new GMeshLib(m_ok, m_analytic);
     373    m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
     374    m_nodelib = new GNodeLib(m_ok, m_analytic, testgeo );
     375 
     376    m_treecheck = new GTreeCheck(m_geolib, m_nodelib, m_ok->getSceneConfig() ) ;
     377 



* analytic is held in GScene



* dont like this split 





How to debug ?
---------------

* investigate the skips (soIdx 27, soIdx 29) 

  * big box with 12 rotated boxes subtracted one by one
  * there is only one each of those meshes (?), so a placeholder for them doesnt explain what is seen
  * dumping the nnode for the polygonization skips shows very big trees  
  * huge boxes with 45 degree rotated boxes subtracted  : they are the near_pool_ows and near_pool_iws
    so they do not explain the missing girders

* DONE : check volume counts, mesh counts and usage totals 

  * rejigged GMeshLib, include MeshUsage.txt with it
   

GDML near_pool_ows0xc2bc1d8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   epsilon:DayaBay_VGDX_20140414-1300 blyth$ cp g4_00.gdml /tmp/


Its a box with 12 rotated boxes subtracted one by one::

     1981     <box lunit="mm" name="near_pool_ows0xc2bc1d8" x="15832" y="9832" z="9912"/>
     1982     <box lunit="mm" name="near_pool_ows_sub00xc55ebf8" x="4179.41484434453" y="4179.41484434453" z="9922"/>
     1983     <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148">
     1984       <first ref="near_pool_ows0xc2bc1d8"/>
     1985       <second ref="near_pool_ows_sub00xc55ebf8"/>
     1986       <position name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148_pos" unit="mm" x="7916" y="4916" z="0"/>
     1987       <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148_rot" unit="deg" x="0" y="0" z="45"/>
     1988     </subtraction>
     1989     <box lunit="mm" name="near_pool_ows_sub10xc21e940" x="4179.41484434453" y="4179.41484434453" z="9922"/>
     1990     <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xc12f640">
     1991       <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c148"/>
     1992       <second ref="near_pool_ows_sub10xc21e940"/>
     1993       <position name="near_pool_ows-ChildFornear_pool_ows_box0xc12f640_pos" unit="mm" x="7916" y="-4916" z="0"/>
     1994       <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xc12f640_rot" unit="deg" x="0" y="0" z="45"/>
     1995     </subtraction>
     .....
     2050     <box lunit="mm" name="near_pool_ows_sub100xbf8c640" x="15824" y="10" z="9912"/>
     2051     <subtraction name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8">
     2052       <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c500"/>
     2053       <second ref="near_pool_ows_sub100xbf8c640"/>
     2054       <position name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8_pos" unit="mm" x="7913" y="0" z="-100"/>
     2055       <rotation name="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8_rot" unit="deg" x="0" y="0" z="90"/>
     2056     </subtraction>
     2057     <box lunit="mm" name="near_pool_ows_sub110xbf8c820" x="15824" y="10" z="9912"/>
     2058     <subtraction name="near_pool_ows_box0xbf8c8a8">
     2059       <first ref="near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8"/>
     2060       <second ref="near_pool_ows_sub110xbf8c820"/>
     2061       <position name="near_pool_ows_box0xbf8c8a8_pos" unit="mm" x="-7913" y="0" z="-100"/>
     2062       <rotation name="near_pool_ows_box0xbf8c8a8_rot" unit="deg" x="0" y="0" z="90"/>
     2063     </subtraction>



soIdx 27 : near_pool_ows0xc2bc1d8_box3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    2018-06-28 13:57:10.865 ERROR [385204] [*X4PhysicalVolume::convertNode@503]  csgnode::dump START for skipped solid soIdx 27
     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)

     du [ 0:bo near_pool_ows0xc2bc1d8_box3] P PRIM  v:0  bb  mi (  -7916.000 -4916.000 -4956.000) mx (   7916.000  4916.000  4956.000) si (  15832.000  9832.000  9912.000)
     gt [ 0:bo near_pool_ows0xc2bc1d8_box3] P NO gtransform 
     du [ 0:bo near_pool_ows_sub00xc55ebf8_box3] P PRIM  v:0  bb  mi (   4960.707  1960.707 -4961.000) mx (  10871.293  7871.293  4961.000) si (   5910.586  5910.586  9922.000)
     gt [ 0:bo near_pool_ows_sub00xc55ebf8_box3] P      gt.t
                0.707   0.707   0.000   0.000 


soIdx 29 : near_pool_iws0xc2cab98_box3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-06-28 13:57:10.914 ERROR [385204] [*X4PhysicalVolume::convertNode@503]  csgnode::dump START for skipped solid soIdx 29
     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:di di] C OPER  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)

     du [ 0:bo near_pool_iws0xc2cab98_box3] P PRIM  v:0  bb  mi (  -6912.000 -3912.000 -4454.000) mx (   6912.000  3912.000  4454.000) si (  13824.000  7824.000  8908.000)
     gt [ 0:bo near_pool_iws0xc2cab98_box3] P NO gtransform 


