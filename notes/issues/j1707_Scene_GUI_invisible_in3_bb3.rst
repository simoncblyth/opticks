FIXED : Scene GUI invisible in3 bb3 (missing struts)
=======================================================


* Scene gui has selector, but seems 4th instance no show ?  Struts are missing 

* Suspect off-by-1 in GScene::makeMergedMeshAndInstancedBuffers() YEP 



::

    op --j1707 --tracer --gltf 3 

    global  :
    bb0     : 36k 3inch PMT bb
    bb1     : 18k 2O inch PMT bb
    bb2     : ~400 fastener? bb
    bb3     : nothing visible ???     <------- ??????
    bb4     : nowt (expected) 
    in0     : 3inch PMT
    in1     : 20 inch PMT
    in2     : the tip of the fastener
    in3     : nowt                    <-------- ????
    in4     : nowt
    axis
    genstep
    nopstep
    photon
    record



Separate renderers for each repeat::

     448 void Scene::initRenderers()
     449 {
     450     LOG(debug) << "Scene::initRenderers "
     451               << " shader_dir " << m_shader_dir
     452               << " shader_incl_path " << m_shader_incl_path
     453                ;
     454   
     455     assert(m_shader_dir);
     456 
     457     m_device = new Device();
     458 
     459     m_colors = new Colors(m_device);
     460 
     461     m_global_renderer = new Renderer("nrm", m_shader_dir, m_shader_incl_path );
     462     m_globalvec_renderer = new Renderer("nrmvec", m_shader_dir, m_shader_incl_path );
     463     m_raytrace_renderer = new Renderer("tex", m_shader_dir, m_shader_incl_path );
     464 
     465    // small array of instance renderers to handle multiple assemblies of repeats 
     466     for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
     467     {
     468         m_instance_mode[i] = false ;
     469         m_instance_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
     470         m_instance_renderer[i]->setInstanced();
     471 
     472         m_bbox_mode[i] = false ;
     473         m_bbox_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
     474         m_bbox_renderer[i]->setInstanced();
     475         m_bbox_renderer[i]->setWireframe(false);  // wireframe is much slower than filled
     476     }




::

     592 void Scene::uploadGeometryInstanced(GMergedMesh* mm)
     593 {
     594     bool empty = mm->isEmpty();
     595     bool skip = mm->isSkip() ;
     596 
     597     if(!skip && !empty)
     598     {
     599 
     600         assert(m_num_instance_renderer < MAX_INSTANCE_RENDERER) ;
     601         LOG(trace)<< "Scene::uploadGeometryInstanced instance renderer " << m_num_instance_renderer  ;
     602 
     603         NPY<float>* ibuf = mm->getITransformsBuffer();
     604         assert(ibuf);
     605 
     606         if(m_instance_renderer[m_num_instance_renderer])
     607         {
     608             m_instance_renderer[m_num_instance_renderer]->upload(mm);
     609             m_instance_mode[m_num_instance_renderer] = true ;
     610         }
     611 
     612         LOG(trace)<< "Scene::uploadGeometryInstanced bbox renderer " << m_num_instance_renderer  ;
     613         GBBoxMesh* bb = GBBoxMesh::create(mm); assert(bb);
     614 
     615         if(m_bbox_renderer[m_num_instance_renderer])
     616         {
     617             m_bbox_renderer[m_num_instance_renderer]->upload(bb);
     618             m_bbox_mode[m_num_instance_renderer] = true ;
     619         }
     620 
     621         m_num_instance_renderer++ ;
     622 
     623     }
     624     else
     625     {
     626          LOG(warning) << "Scene::uploadGeometry SKIPPING "
     627                       << " empty " << empty
     628                       << " skip " << skip
     629                       ;
     630     }
     631 }


::

    op --j1707 --tracer --gltf 3 --OGLRAP trace
    op --j1707 --tracer --gltf 3 --OGLRAP debug



Should have 5 mm, with global in mm0, but only see 4::

    2017-08-17 16:46:13.522 DEBUG [261798] [Scene::uploadGeometry@640] Scene::uploadGeometry nmm 4
    2017-08-17 16:46:13.522 DEBUG [261798] [Scene::uploadGeometryGlobal@565] Scene::uploadGeometryGlobal 
    2017-08-17 16:46:13.522 DEBUG [261798] [Scene::uploadGeometry@650] Scene::uploadGeometry 0 geoCode A
    2017-08-17 16:46:13.553 DEBUG [261798] [Scene::uploadGeometry@650] Scene::uploadGeometry 1 geoCode A
    2017-08-17 16:46:13.574 DEBUG [261798] [Scene::uploadGeometry@650] Scene::uploadGeometry 2 geoCode A
    2017-08-17 16:46:13.593 DEBUG [261798] [Scene::uploadGeometry@650] Scene::uploadGeometry 3 geoCode A 
    2017-08-17 16:46:13.612 DEBUG [261798] [Scene::uploadGeometry@664] Scene::uploadGeometry m_num_instance_renderer 3




::

    242 void OpticksViz::uploadGeometry()
    243 {
    244     NPY<unsigned char>* colors = m_hub->getColorBuffer();
    245 
    246     m_scene->uploadColorBuffer( colors );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"
    247 
    248     LOG(info) << m_ok->description();
    249 
    250     m_composition->setTimeDomain(        m_ok->getTimeDomain() );
    251     m_composition->setDomainCenterExtent(m_ok->getSpaceDomain());
    252 
    253     m_scene->setGeometry(m_hub->getGGeo());
    254 
    255     m_scene->uploadGeometry();
    256 
    257     bool autocam = true ;
    258 
    259     // handle commandline --target option that needs loaded geometry 
    260     unsigned int target = m_geometry->getTargetDeferred();   // default to 0 
    261     LOG(debug) << "App::uploadGeometryViz setting target " << target ;
    262 
    263     m_geometry->setTarget(target, autocam);
    264 
    265 }


::

     634 void Scene::uploadGeometry()
     635 {
     636     // currently invoked from ggeoview main
     637     assert(m_ggeo && "must setGeometry first");
     638     unsigned int nmm = m_ggeo->getNumMergedMesh();
     639 
     640     LOG(debug) << "Scene::uploadGeometry"
     641               << " nmm " << nmm
     642               ;



5 triangulated are loaded::

    2017-08-17 16:56:40.280 INFO  [269873] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-17 16:56:40.523 INFO  [269873] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-17 16:56:40.565 INFO  [269873] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/2 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/2 index 2 version (null) existsdir 1
    2017-08-17 16:56:40.590 INFO  [269873] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/3 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/3 index 3 version (null) existsdir 1
    2017-08-17 16:56:40.591 INFO  [269873] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/4 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/4 index 4 version (null) existsdir 1
    2017-08-17 16:56:41.162 INFO  [269873] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae



But GScene only sees 4::

    2017-08-17 16:57:13.796 INFO  [269873] [GScene::init@165] GScene::init createVolumeTrue selected_count 290276
    2017-08-17 16:57:14.541 INFO  [269873] [GScene::makeMergedMeshAndInstancedBuffers@917] GScene::makeMergedMeshAndInstancedBuffers num_repeats 4 START 
    2017-08-17 16:57:27.763 INFO  [269873] [GScene::makeMergedMeshAndInstancedBuffers@971] GScene::makeMergedMeshAndInstancedBuffers DONE num_repeats 4 nmm_created 4 nmm 4
    2017-08-17 16:57:27.764 INFO  [269873] [GScene::prepareVertexColors@204] GScene::prepareVertexColors START




::

     911 void GScene::makeMergedMeshAndInstancedBuffers()   // using m_geolib to makeMergedMesh
     912 {
     913     unsigned num_repeats = std::max<unsigned>(1u,m_scene->getNumRepeats()); // global 0 included
     914     unsigned nmm_created = 0 ;
     915 
     916     if(m_verbosity > 0)
     917     LOG(info) << "GScene::makeMergedMeshAndInstancedBuffers num_repeats " << num_repeats << " START " ;
     918 


::

    1453 unsigned NScene::getRepeatCount(unsigned ridx)
    1454 {
    1455     return m_repeat_count[ridx] ;
    1456 }
    1457 unsigned NScene::getNumRepeats()
    1458 {
    1459    // this assumes ridx is a contiguous index
    1460     return m_repeat_count.size() ;
    1461 }


    1371 void NScene::labelTree()
    1372 {
    1373     for(unsigned i=0 ; i < m_repeat_candidates.size() ; i++)
    1374     {
    1375          std::string pdig = m_repeat_candidates.at(i);
    1376 
    1377          unsigned ridx = deviseRepeatIndex(pdig);
    1378 
    1379          assert(ridx == i + 1 );
    1380 
    1381          std::vector<nd*> instances = m_root->find_nodes(pdig);
    1382 
    1383          // recursive labelling starting from the instances
    1384          for(unsigned int p=0 ; p < instances.size() ; p++)
    1385          {
    1386              labelTree_r(instances[p], ridx);
    1387          }
    1388     }
    1389 
    1390     //if(m_verbosity > 1)
    1391     LOG(info)<<"NScene::labelTree" 
    1392              << " label_count (non-zero ridx labelTree_r) " << m_label_count 
    1393              << " num_repeat_candidates " << m_repeat_candidates.size()
    1394              ;
    1395 }



    1394 #ifdef OLD_LABEL_TREE
    1395 void NScene::labelTree_r(nd* n, unsigned /*ridx*/)
    1396 {
    1397     unsigned ridx = deviseRepeatIndex_0(n) ;
    1398 #else
    1399 void NScene::labelTree_r(nd* n, unsigned ridx)
    1400 {
    1401 #endif
    1402     n->repeatIdx = ridx ;
    1403 
    1404     if(m_repeat_count.count(ridx) == 0) m_repeat_count[ridx] = 0 ;
    1405     m_repeat_count[ridx]++ ;
    1406 
    1407     if(ridx > 0) m_label_count++ ;
    1408 
    1409     for(nd* c : n->children) labelTree_r(c, ridx) ;
    1410 }
    1411 





