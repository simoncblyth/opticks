AssimpNode copyMeshes causing memory issue with JUNO geometry
================================================================

* linux with JUNO geometry takes a long time to create the 
  geocache and pushes to limits of available memory

* initially thought this was due to premature globablization 
  of verts : and hence easy to avoid by deferring this, 
  but looking more closely seems not so easy 

* BUT: actually could avoid mesh copying by operating direct 
  from raw meshes... and hence cut memory requirements in half



Can the copy be skippe
--------------------------

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
     159 



::


    193 void AssimpImporter::import(unsigned int flags)
    194 {
    195     LOG(info) << "AssimpImporter::import path " << m_path << " flags " << flags ;
    196     m_process_flags = flags ;
    197 
    198     assert(m_path);
    199     m_aiscene = m_importer->ReadFile( m_path, flags );
    200 
    201     if(!m_aiscene)
    202     {
    203         printf("AssimpImporter::import ERROR : \"%s\" \n", m_importer->GetErrorString() );
    204         return ;
    205     }
    206 
    207     //dumpProcessFlags("AssimpImporter::import", flags);
    208     //dumpSceneFlags("AssimpImporter::import", m_aiscene->mFlags);
    209 
    210     Summary("AssimpImporter::import DONE");
    211 
    212     m_tree = new AssimpTree(m_aiscene);
    213 }


    016 AssimpTree::AssimpTree(const aiScene* scene)
     17   :
     18   m_scene(scene),
     19   m_root(NULL),
     20   m_registry(NULL),
     21   m_index(0),
     22   m_raw_index(0),
     23   m_wrap_index(0),
     24   m_dump(0)
     25 {
     26    m_registry = new AssimpRegistry ;
     27 
     28    traverseWrap();
     29 
     30    //m_registry->summary();
     31 }



    043 void AssimpTree::traverseWrap(const char* msg)
     44 {
     45    LOG(debug) << msg ;
     46 
     47    m_wrap_index = 0 ;
     48    m_dump = 0 ;
     49 
     50    std::vector<aiNode*> ancestors ;
     51 
     52    traverseWrap(m_scene->mRootNode, ancestors);
     53 
     54    LOG(debug) << "AssimpTree::traverseWrap m_wrap_index " << m_wrap_index << " m_dump " << m_dump ;
     55 }


    057 void AssimpTree::traverseWrap(aiNode* node, std::vector<aiNode*> ancestors)
     58 {
     59    //
     60    // every node of the tree needs its own nodepath vector
     61    // this is used to establish a digest for each node, and 
     62    // a pdigest for the parent 
     63    //
     64    // NB the nodepath is complete, ie agnostic regarding visit criteria
     65 
     66    std::vector<aiNode*> nodepath = ancestors ;
     67    nodepath.push_back(node) ;
     68 
     69    if(node->mNumMeshes > 0) visitWrap(nodepath);
     70 
     71    for(unsigned int i = 0; i < node->mNumChildren; i++) traverseWrap(node->mChildren[i], nodepath);
     72 }




    075 void AssimpTree::visitWrap(std::vector<aiNode*> nodepath)
     76 {
     77    AssimpNode* wrap = new AssimpNode(nodepath, this) ;
     78 
     79    wrap->setIndex(m_wrap_index);
     80 
     81    if(m_wrap_index == 0) setRoot(wrap);
     82 
     83    m_registry->add(wrap);
     84 
     85    // hookup relationships via digest matching : works on 1st pass as 
     86    // parents always exist before children 
     87    AssimpNode* parent = m_registry->lookup(wrap->getParentDigest());
     88 
     89    if(parent)
     90    {
     91        wrap->setParent(parent);
     92        parent->addChild(wrap);
     93    }
     94 
     95    aiMatrix4x4 transform = wrap->getGlobalTransform() ;
     96    wrap->copyMeshes(transform);
     97 
     98    //if(m_wrap_index == 5000) wrap->ancestors();
     99 
    100 
    101    if(0)
    102    {
    103        if(parent) parent->summary("AssimpTree::traW--parent");
    104        wrap->summary("AssimpTree::traverseWrap");
    105        dumpTransform("AssimpTree::traverseWrap transform", transform);
    106    }
    107 
    108    m_wrap_index++;
    109 }






Looking at the users of AssimpTree it appears the globaliz mesh in the noe is not use


::

     723 GMesh* AssimpGGeo::convertMesh(unsigned int index )
     724 {
     725     const aiScene* scene = m_tree->getScene();
     726     assert(index < scene->mNumMeshes);
     727     aiMesh* mesh = scene->mMeshes[index] ;
     728     GMesh* graw = convertMesh(mesh, index );
     729     return graw ;
     730 }
     731 
     732 
     733 void AssimpGGeo::convertMeshes(const aiScene* scene, GGeo* gg, const char* /*query*/)
     734 {
     735     LOG(info)<< "AssimpGGeo::convertMeshes NumMeshes " << scene->mNumMeshes ;
     736 
     737     for(unsigned int i = 0; i < scene->mNumMeshes; i++)
     738     {
     739         aiMesh* mesh = scene->mMeshes[i] ;
     740    
     741         const char* meshname = mesh->mName.C_Str() ;
     742 
     743         GMesh* graw = convertMesh(mesh, i );
     744 
     745         GMesh* gmesh = graw->makeDedupedCopy(); // removes duplicate vertices, re-indexing faces accordingly
     746 
     747         delete graw ;
     748 
     749         gmesh->setName(meshname);
     750 
     751         GMesh* gfixed = gg->invokeMeshJoin(gmesh);
     752 
     753         assert(gfixed) ;
     754 
     755         if(gfixed != gmesh)
     756         {
     757             LOG(trace) << "AssimpGGeo::convertMeshes meshfixing was done for "
     758                         << " meshname " << meshname
     759                         << " index " << i
     760                          ;
     761 
     762             delete gmesh ;
     763         }
     764 
     765         gg->add(gfixed);
     766 
     767     }
     768 }

