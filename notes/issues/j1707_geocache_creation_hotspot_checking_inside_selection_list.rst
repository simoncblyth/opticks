FIXED : Geocache creation hotspot checking node in selection list
==================================================================

* but normally the selection is ALL, so special case this fixes hotspot 


Making geocache
-----------------

::

    simon:issues blyth$ op --j1707 -G


hotspot AssimpSelection::contains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (lldb) bt
    * thread #1: tid = 0x2a1de, 0x0000000101e08336 libAssimpRap.dylib`AssimpSelection::contains(AssimpNode*) [inlined] bool std::__1::operator==<AssimpNode**, AssimpNode**>(__x=0x00007fff5fbfcb18, __y=0x00007fff5fbfcb10) + 15 at iterator:1287, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
      * frame #0: 0x0000000101e08336 libAssimpRap.dylib`AssimpSelection::contains(AssimpNode*) [inlined] bool std::__1::operator==<AssimpNode**, AssimpNode**>(__x=0x00007fff5fbfcb18, __y=0x00007fff5fbfcb10) + 15 at iterator:1287


::

    (lldb) f 14
    frame #14: 0x0000000101e1060c libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007fff5fbfd5b0, ctrl=0x00007fff5fbff00d) + 380 at AssimpGGeo.cc:170
       167      convertMaterials(scene, m_ggeo, ctrl );
       168      convertSensors( m_ggeo ); 
       169      convertMeshes(scene, m_ggeo, ctrl);
    -> 170      convertStructure(m_ggeo);
       171  
       172      return 0 ;
       173  }


       817         GSolid* solid = convertStructureVisit( gg, node, depth, parent);
       818  
    -> 819      bool selected = m_selection && m_selection->contains(node) ;  
       820  
       821      solid->setSelected(selected);


       97      bool AssimpSelection::contains(AssimpNode* node)
       98   {
    -> 99       return std::find(m_selection.begin(), m_selection.end(), node ) != m_selection.end() ;  
       100  }



Hmm, probably much faster to re-apply a selection rather than checking 
for the node in the list of 0.25M nodes.
But normally the selection is ALL, so can just special case this.


::

     119 int AssimpGGeo::load(GGeo* ggeo)
     120 {
     121     // THIS IS THE ENTRY POINT SET IN OpticksGeometry::loadGeometryBase
     122 
     123     Opticks* opticks = ggeo->getOpticks();
     124     OpticksResource* resource = opticks->getResource();
     125     OpticksQuery* query = opticks->getQuery() ;
     126 
     127     const char* path = opticks->getDAEPath() ;
     128     const char* ctrl = resource->getCtrl() ;
     129     unsigned int verbosity = ggeo->getLoaderVerbosity();
     130 
     131     LOG(info)<< "AssimpGGeo::load "
     132              << " path " << ( path ? path : "NULL" )
     133              << " query " << ( query ? query->getQueryString() : "NULL" )
     134              << " ctrl " << ( ctrl ? ctrl : "NULL" )
     135              << " verbosity " << verbosity
     136              ;
     137 
     138     assert(path);
     139     assert(query);
     140     assert(ctrl);
     141 
     142     AssimpImporter assimp(path);
     143 
     144     assimp.import();
     145 
     146     AssimpSelection* selection = assimp.select(query);
     147 
     148     AssimpTree* tree = assimp.getTree();
     149 
     150 
     151     AssimpGGeo agg(ggeo, tree, selection);
     152 
     153     agg.setVerbosity(verbosity);
     154 
     155     int rc = agg.convert(ctrl);
     156 
     157     return rc ;
     158 }





