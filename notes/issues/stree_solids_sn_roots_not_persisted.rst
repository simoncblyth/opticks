stree_solids_sn_roots_not_persisted
======================================




stree.h::

     332     std::vector<std::string> bdname ;
     333     std::vector<std::string> implicit ;  // names of implicit surfaces
     334 
     335     std::vector<std::string> soname_raw ;   // solid names, my have 0x pointer suffix 
     336     std::vector<std::string> soname ;       // unique solid names, created with sstr::StripTail_unique with _1 _2 ... uniqing 
     337     std::vector<sn*>         solids ;       // used from U4Tree::initSolid but not currently persisted 
     338 
     339     std::vector<glm::tmat4x4<double>> m2w ; // local (relative to parent) "model2world" transforms for all nodes
     340     std::vector<glm::tmat4x4<double>> w2m ; // local (relative to parent( "world2model" transforms for all nodes  
     341     std::vector<glm::tmat4x4<double>> gtd ; // global (relative to root) "GGeo Transform Debug" transforms for all nodes
     342     // "gtd" formerly from X4PhysicalVolume::convertStructure_r
     343 
     344     std::vector<snode> nds ;               // snode info for all structural nodes, the volumes
     345     std::vector<snode> rem ;               // subset of nds with the remainder nodes
     346     std::vector<snode> tri ;               // subset of nds which are configured to be force triangulated (expected to otherwise be remainder nodes)
     347     std::vector<std::string> digs ;        // per-node digest for all nodes  
     348     std::vector<std::string> subs ;        // subtree digest for all nodes
     349     std::vector<sfactor> factor ;          // small number of unique subtree factor, digest and freq  
     350 
     351     std::vector<int> sensor_id ;           // updated by reorderSensors
     352     unsigned sensor_count ;
     353     std::vector<std::string> sensor_name ;
     354 
     355 
     356     sfreq* subs_freq ;                     // occurence frequency of subtree digests in entire tree 
     357                                            // subs are collected in stree::classifySubtrees
     358 
     359     s_csg* _csg ;                          // sn.h based csg node trees
     360 



::

     629 inline void U4Tree::initSolid(const G4VSolid* const so, int lvid )
     630 {
     631     G4String _name = so->GetName() ; // bizarre: G4VSolid::GetName returns by value, not reference
     632     const char* name = _name.c_str();
     633 
     634     assert( int(solids.size()) == lvid );
     635     int d = 0 ;
     636     sn* root = U4Solid::Convert(so, lvid, d );
     637     assert( root );
     638 
     639     solids.push_back(so);
     640     st->soname_raw.push_back(name);
     641     st->solids.push_back(root);
     642 
     643 }



