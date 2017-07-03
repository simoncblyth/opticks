Geoselection at py level inconvenient
========================================


* Means must keep regenerating GLTF... as change selection.

Moving to GScene::createVolumeTree_r would allow reuse of OpticksQuery 





::

    375 GSolid* GScene::createVolumeTree_r(nd* n, GSolid* parent)

::

     806 void AssimpGGeo::convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
     807 {
     808     // recursive traversal of the AssimpNode tree
     809     // note that full tree is traversed even when a partial selection is applied 
     810 
     811 
     812     GSolid* solid = convertStructureVisit( gg, node, depth, parent);
     813 
     814     bool selected = m_selection && m_selection->contains(node) ;
     815 
     816     solid->setSelected(selected);
     817 

