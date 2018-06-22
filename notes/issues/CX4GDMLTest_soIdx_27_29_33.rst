CX4GDMLTest_soIdx_27_29_33
=============================


Observations on CX4GDML:

* suspect time consuming CSG node nudging is happening at node level when 
  it could be done at mesh level : this would save loada time, by only 
  doing this once per shape



Skipped three big CSG trees::

    418      Mh* mh = m_sc->get_mesh_for_node( ndIdx );  // node->mesh via soIdx (the local mesh index)
    419      
    420      std::vector<unsigned> skips = {27, 29, 33 };
    421      
    422      if(mh->csg == NULL)
    423      {   
    424          //convertSolid(mh, solid);
    425          mh->csg = X4Solid::Convert(solid) ;  // soIdx 33 giving analytic problems too 
    426          
    427          bool placeholder = std::find( skips.begin(), skips.end(), nd->soIdx ) != skips.end()  ;
    428 
    429          mh->mesh = placeholder ? X4Mesh::Placeholder(solid) : X4Mesh::Convert(solid) ;
    430       
    431          mh->vtx = mh->mesh->m_x4src_vtx ;
    432          mh->idx = mh->mesh->m_x4src_idx ;
    433      }
    434 



::

    36527 2018-06-21 21:09:29.121 INFO  [24382395] [*X4PhysicalVolume::convertNode@404] convertNode  ndIdx 3150 soIdx 27 lvIdx 232 soName near_pool_ows_box0xbf8c8a8
    36528 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name               near_pool_ows_box0xbf8c8a8 entityType 3 entityName G4SubtractionSolid root 0x0
    36529 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xbf8c6c8 entityType 3 entityName G4SubtractionSolid root 0x0
    36530 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xbf8c500 entityType 3 entityName G4SubtractionSolid root 0x0
    36531 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc356df8 entityType 3 entityName G4SubtractionSolid root 0x0
    36532 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc2c4a40 entityType 3 entityName G4SubtractionSolid root 0x0
    36533 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc21d530 entityType 3 entityName G4SubtractionSolid root 0x0
    36534 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc12e148 entityType 3 entityName G4SubtractionSolid root 0x0
    36535 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xbf97a68 entityType 3 entityName G4SubtractionSolid root 0x0
    36536 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc12de98 entityType 3 entityName G4SubtractionSolid root 0x0
    36537 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc357900 entityType 3 entityName G4SubtractionSolid root 0x0
    36538 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xc12f640 entityType 3 entityName G4SubtractionSolid root 0x0
    36539 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name near_pool_ows-ChildFornear_pool_ows_box0xbf8c148 entityType 3 entityName G4SubtractionSolid root 0x0
    36540 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                   near_pool_ows0xc2bc1d8 entityType 5 entityName G4Box root 0x0
    36541 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36542 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub00xc55ebf8 entityType 5 entityName G4Box root 0x0
    36543 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36544 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub10xc21e940 entityType 5 entityName G4Box root 0x0
    36545 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36546 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub20xc2344b0 entityType 5 entityName G4Box root 0x0
    36547 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36548 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub30xbf5f5b8 entityType 5 entityName G4Box root 0x0
    36549 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36550 2018-06-21 21:09:29.121 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub40xbf979e0 entityType 5 entityName G4Box root 0x0
    36551 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36552 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub50xc12e0c0 entityType 5 entityName G4Box root 0x0
    36553 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36554 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub60xc2a23c8 entityType 5 entityName G4Box root 0x0
    36555 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36556 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub70xc21d660 entityType 5 entityName G4Box root 0x0
    36557 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36558 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub80xc2c4b70 entityType 5 entityName G4Box root 0x0
    36559 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36560 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name              near_pool_ows_sub90xc356f50 entityType 5 entityName G4Box root 0x0
    36561 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36562 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name             near_pool_ows_sub100xbf8c640 entityType 5 entityName G4Box root 0x0
    36563 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name                                  placedB entityType 0 entityName G4DisplacedSolid root 0x0
    36564 2018-06-21 21:09:29.122 INFO  [24382395] [X4Solid::init@57] X4SolidBase name             near_pool_ows_sub110xbf8c820 entityType 5 entityName G4Box root 0x0
    36565 2018-06-21 21:09:29.122 INFO  [24382395] [*X4Mesh::Placeholder@31]  visExtent G4VisExtent (bounding box):
    36566   X limits: -7916 7916
    36567   Y limits: -4916 4916
    36568   Z limits: -4956 4956
    36569 2018-06-21 21:09:29.122 INFO  [24382395] [NNodeNudger::update_prim_bb@37] NNodeNudger::update_prim_bb nprim 13



