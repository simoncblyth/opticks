Too Many Ridx Instances
=========================


Initial simple instancing criteria from NScene::labelTree_r of "num_mesh_instances > 4" 
is yielding 117 distinct types of geometry, that is an order of magnitude more than desired.

The very simple instancing criteria based on mesh instance counts alone 
yields many more instances than optimal as it is unaware of the containment relations 
within the node tree. 

Using containment relations allow "vertical" combination in the node tree not just 
the current "horizontal" combinations. 

NScene now does repeat candidate finding similar to GTreeCheck/GNode progeny digests within NScene. 



Initial Approach
-------------------

::

    252 unsigned NScene::deviseRepeatIndex(nd* n)
    253 {
    254     unsigned mesh_idx = n->mesh ;
    255     unsigned num_mesh_instances = getNumInstances(mesh_idx) ;
    256 
    257     unsigned ridx = 0 ;   // <-- global default ridx
    258 
    259     bool make_instance  = num_mesh_instances > 4  ;
    260 
    261     if(make_instance)
    262     {
    263         if(m_mesh2ridx.count(mesh_idx) == 0)
    264              m_mesh2ridx[mesh_idx] = m_mesh2ridx.size() + 1 ;
    265 
    266         ridx = m_mesh2ridx[mesh_idx] ;
    267 
    268         // ridx is a 1-based contiguous index tied to the mesh_idx 
    269         // using trivial things like "mesh_idx + 1" causes  
    270         // issue downstream which expects a contiguous range of ridx 
    271         // when using partial geometries 
    272     }
    273     return ridx ;
    274 }
    275 
    276 void NScene::labelTree_r(nd* n)
    277 {
    278     unsigned ridx = deviseRepeatIndex(n);
    279 
    280     n->repeatIdx = ridx ;
    281 
    282     if(m_repeat_count.count(ridx) == 0) m_repeat_count[ridx] = 0 ;
    283     m_repeat_count[ridx]++ ;
    284 
    285 
    286     for(nd* c : n->children) labelTree_r(c) ;
    287 }





::

    tgltf-;tgltf-gdml

    2017-05-24 12:22:23.820 INFO  [2974756] [*GScene::createVolumeTree@131] GScene::createVolumeTree DONE num_nodes: 12229
    2017-05-24 12:22:23.851 INFO  [2974756] [GScene::makeMergedMeshAndInstancedBuffers@269] GScene::makeMergedMeshAndInstancedBuffers num_repeats 117 START 
    2017-05-24 12:22:54.614 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 1
    2017-05-24 12:22:54.683 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 1
    2017-05-24 12:22:55.255 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 11
    2017-05-24 12:22:55.334 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 15
    2017-05-24 12:22:55.483 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 33
    2017-05-24 12:22:56.197 INFO  [2974756] [GScene::makeMergedMeshAndInstancedBuffers@319] GScene::makeMergedMeshAndInstancedBuffers DONE num_repeats 117 nmm_created 117 nmm 117
    Assertion failed: (0 && "early exit for gltf==4"), function loadFromGLTF, file /Users/blyth/opticks/ggeo/GGeo.cc, line 660.




GTreeCheck triangulated approach
-----------------------------------------


::

    027 GTreeCheck::GTreeCheck(GGeo* ggeo)
     28        :
     29        m_ggeo(ggeo),
     30        m_geolib(ggeo->getGeoLib()),
     31        m_repeat_min(120),
     32        m_vertex_min(300),   // aiming to include leaf? sStrut and sFasteners
     33        m_root(NULL),
     34        m_count(0),
     35        m_labels(0),
     36        m_digest_count(new Counts<unsigned>("progenyDigest"))
     37 {
     38 }


     87 void GTreeCheck::traverse()
     88 {
     89     m_root = m_ggeo->getSolid(0);
     90     assert(m_root);
     91 
     92     // count occurences of distinct progeny digests (relative sub-tree identities) in m_digest_count 
     93     traverse_r(m_root, 0);
     94 
     95     m_digest_count->sort(false);   // descending count order, ie most common subtrees first
     96     //m_digest_count->dump();
     97 
     98     // minrep 120 removes repeats from headonPMT, calibration sources and RPC leaving just PMTs 
     99 
    100     // collect digests of repeated pieces of geometry into  m_repeat_candidates
    101     findRepeatCandidates(m_repeat_min, m_vertex_min);
    102     dumpRepeatCandidates();
    103 }
    104 
    105 void GTreeCheck::traverse_r( GNode* node, unsigned int depth)
    106 {
    107     std::string& pdig = node->getProgenyDigest();
    108     m_digest_count->add(pdig.c_str());
    109     m_count++ ;
    110 
    111     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1 );
    112 }


    155 void GTreeCheck::findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min)
    156 {
    157     unsigned int nall = m_digest_count->size() ;
    ...
    166     // over distinct subtrees (ie progeny digests)
    167     for(unsigned int i=0 ; i < nall ; i++)
    168     {
    169         std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;
    170 
    171         std::string& pdig = kv.first ;
    172         unsigned int ndig = kv.second ;                 // number of occurences of the progeny digest 
    173 
    174         GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    175 
    176         // suspect problem with allowing leaf repeaters is that digesta are not-specific enough, 
    177         // so get bad matching 
    178         //
    179         //  allowing leaf repeaters results in too many, so place vertex count reqirement too 
    180 
    181 
    182         unsigned int nprog = node->getProgenyCount() ;  // includes self when GNode.m_selfdigest is true
    183         unsigned int nvert = node->getProgenyNumVertices() ;  // includes self when GNode.m_selfdigest is true
    184 
    185        // hmm: maybe selecting based on  ndig*nvert 
    186        // but need to also require ndig > smth as dont want to repeat things like the world 
    187 
    188         bool select = ndig > repeat_min && nvert > vertex_min ;
    189 
    190         if(i < 15) LOG(info)
    191                   << ( select ? "**" : "  " )
    192                   << " i "     << std::setw(3) << i
    193                   << " pdig "  << std::setw(32) << pdig
    194                   << " ndig "  << std::setw(6) << ndig
    195                   << " nprog " <<  std::setw(6) << nprog
    196                   << " nvert " <<  std::setw(6) << nvert
    197                   << " n "     <<  node->getName()
    198                   ;
    199 
    200         if(select) m_repeat_candidates.push_back(pdig);
    201     }
    202 
    203     // erase repeats that are enclosed within other repeats 
    204     // ie that have an ancestor which is also a repeat candidate
    205 
    206     m_repeat_candidates.erase(
    207          std::remove_if(m_repeat_candidates.begin(), m_repeat_candidates.end(), *this ),
    208          m_repeat_candidates.end()
    209     );
    210 
    211 
    212 }
    213 
    214 bool GTreeCheck::operator()(const std::string& dig)
    215 {
    216     bool cr = isContainedRepeat(dig, 3);
    217 
    218     if(cr) LOG(info)
    219                   << "GTreeCheck::operator() "
    220                   << " pdig "  << std::setw(32) << dig
    221                   << " disallowd as isContainedRepeat "
    222                   ;
    223 
    224     return cr ;
    225 }
    226 
    227 bool GTreeCheck::isContainedRepeat( const std::string& pdig, unsigned int levels ) const
    228 {
    229     // for the first node that matches the *pdig* progeny digest
    230     // look back *levels* ancestors to see if any of the immediate ancestors 
    231     // are also repeat candidates, if they are then this is a contained repeat
    232     // and is thus disallowed in favor of the ancestor that contains it 
    233 
    234     GNode* node = m_root->findProgenyDigest(pdig) ;
    235     std::vector<GNode*>& ancestors = node->getAncestors();
    236     unsigned int asize = ancestors.size();
    237 
    238     for(unsigned int i=0 ; i < std::min(levels, asize) ; i++)
    239     {
    240         GNode* a = ancestors[asize - 1 - i] ;
    241         std::string& adig = a->getProgenyDigest();
    242         if(std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), adig ) != m_repeat_candidates.end())
    243         {
    244             return true ;
    245         }
    246     }
    247     return false ;
    248 }




    015 class GGEO_API GNode {
    ...
    148   private:
    149       std::string         m_local_digest ;
    150       std::string         m_progeny_digest ;
    151       std::vector<GNode*> m_progeny ;
    152       std::vector<GNode*> m_ancestors ;

    024 GNode::GNode(unsigned int index, GMatrixF* transform, GMesh* mesh)
     25     :
     26     m_selfdigest(true),


    442 std::string& GNode::getProgenyDigest()
    443 {
    444     if(m_progeny_digest.empty())
    445     {
    446         std::vector<GNode*>& progeny = getProgeny();
    447         m_progeny_count = progeny.size();
    448         GNode* extra = m_selfdigest ? this : NULL ;
    449         m_progeny_digest = GNode::localDigest(progeny, extra) ;
    450     }
    451     return m_progeny_digest ;
    452 }

    283 std::vector<GNode*>& GNode::getProgeny()
    284 {
    285     if(m_progeny.size() == 0)
    286     {
    287         // call on children, as wish to avoid collecting self  
    288         for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(m_progeny); 
    289     }
    290     return m_progeny ; 
    291 }
    292 
    293 void GNode::collectProgeny(std::vector<GNode*>& progeny)
    294 {
    295     progeny.push_back(this);
    296     for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(progeny);
    297 }


