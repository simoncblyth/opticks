j1707_250k_volumes_render_slowly_with_full_tri
===================================================

Issue
--------






Huge geometry, > 250k Volumes, 18k instances + 38k instances

* performant in bbox mode, 60-30 fps gives nice interactivity
* (hit B key to shift mode) bogs down in full tri mode ~3 fps gives painful interactivity

This is without any culling or LOD.




Thoughts
-------------

* LOD is a global thing, do not need individual solid control, 
  so use standard OpticksCfg to configure/control via --lod N --lodconfig "levels=2"

* each CSG tree yields a GSolid, with associated GParts

* the GSolids are combined to form the GMergedMesh 



Test Commands
-------------------

::


    op --j1707 --gltf 3 --tracer --instcull

        ## currently just tests that InstLODCull shader compiles 


    tboolean-;tboolean-torus --lod 1 --lodconfig "levels=3,verbosity=2" --debugger 

        ## check LODification and rendering of test geometry   


LODCull for all instances or just the PMTs ? JUST PMTs
-----------------------------------------------------------

* switching between bbox and inst rendering for the sFasteners and sStrut 
  makes no difference to interactivity... 

* almost certainly applying LODCull to only 480 instances would not be beneficial : 
  just not enough instances to see any benefit, using INSTANCE_MINIMUM = 10000
  

::

    // :set nowrap

   2017-08-28 11:52:34.748 INFO  [1181753] [NScene::dumpRepeatCount@1429] NScene::dumpRepeatCount m_verbosity 1
     ridx   1 count 182860   ## 36572*(4+1) = 182860   PMT_3inch_pmt_solid0x1c9e270    (progeny 4)
     ridx   2 count 106434   ## 17739*(5+1) = 106434   sMask_virtual0x18163c0          (progeny 5) 
     ridx   3 count   480    ##   480*(0+1) =    480   sFasteners0x1506180             (progeny 0)
     ridx   4 count   480    ##   480*(0+1) =    480   sStrut0x14ddd50                 (progeny 0)

     **  ##  idx   0 pdig 68a31892bccd1741cc098d232c702605 num_pdig  36572 num_progeny      4 NScene::meshmeta mesh_id  22 lvidx  20 height  1 soname        PMT_3inch_pmt_solid0x1c9e270 lvname              PMT_3inch_log0x1c9ef80
     **      idx   1 pdig 683529bb1b0fedc340f2ebce47468395 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  26 lvidx  19 height  0 soname       PMT_3inch_cntr_solid0x1c9e640 lvname         PMT_3inch_cntr_log0x1c9f1f0
     **      idx   2 pdig c81fb13777b701cb8ce6cdb7f0661f1b num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  25 lvidx  17 height  0 soname PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 lvname       PMT_3inch_inner2_log0x1c9f120
     **      idx   3 pdig 83a5a282f092aa7baf6982b54227bb54 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  24 lvidx  16 height  0 soname PMT_3inch_inner1_solid_ell_helper0x1c9e510 lvname       PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 50308873a9847d1c2c2029b6c9de7eeb num_pdig  36572 num_progeny      2 NScene::meshmeta mesh_id  23 lvidx  18 height  0 soname PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0 lvname         PMT_3inch_body_log0x1c9eef0
     **      idx   5 pdig 27a989a1aeab2b96cedd2b6c4a7cba2f num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  17 lvidx  10 height  2 soname                      sMask0x1816f50 lvname                      lMask0x18170e0
     **      idx   6 pdig e39a411b54c3ce46fd382fef7f632157 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  21 lvidx  12 height  4 soname    PMT_20inch_inner2_solid0x1863010 lvname      PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 74d8ce91d143cad52fad9d3661dded18 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  20 lvidx  11 height  4 soname    PMT_20inch_inner1_solid0x1814a90 lvname      PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig a80803364fbf92f1b083ebff420b6134 num_pdig  17739 num_progeny      2 NScene::meshmeta mesh_id  19 lvidx  13 height  3 soname      PMT_20inch_body_solid0x1813ec0 lvname        PMT_20inch_body_log0x1863160
     **      idx   9 pdig 6b1283d04ffc8a27e19f84e2bec2ddd6 num_pdig  17739 num_progeny      3 NScene::meshmeta mesh_id  18 lvidx  14 height  3 soname       PMT_20inch_pmt_solid0x1813600 lvname             PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig 8cbe68d7d5c763820ff67b8088e0de98 num_pdig  17739 num_progeny      5 NScene::meshmeta mesh_id  16 lvidx  15 height  0 soname              sMask_virtual0x18163c0 lvname               lMaskVirtual0x1816910
     **  ##  idx  11 pdig ad8b68a55505a09ac7578f32418904b3 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  15 lvidx   9 height  2 soname                 sFasteners0x1506180 lvname                 lFasteners0x1506370
     **  ##  idx  12 pdig f93b8bbbac89ea22bac0bf188ba49a61 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  14 lvidx   8 height  1 soname                     sStrut0x14ddd50 lvname                     lSteel0x14dde40




How to integrate something like env-/instcull-/LODCullShader into oglrap ?
----------------------------------------------------------------------------

Differences, 

* UBO rather than lots of little uniform calls


LODCullShader via transform feedback and geometry shader forks an original 
instance transforms buffer into three separate GPU buffers (for three LOD levels), 
filtering by instance center positions being within frustum of current view and forking 
by distance from the eye to the instances into 3 LOD piles.


How to structure ?
~~~~~~~~~~~~~~~~~~~~~~

* LODCull needs to be an optional constituent of the instanced oglrap-/Renderer 
  depending on instance transform counts exceeding a minimum as configured in oglrap-/Scene


DONE : LODify GMergedMesh 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm similar to the below but need to retain the offsets for each component of the LOD,
to allow drawing them individually.

::

    GMergedMesh* GMergedMesh::combine 
    void GMergedMesh::mergeMergedMesh


Hmm need sidecar NPY<int> buffer to hold the offsets...


::

    127 void ICDemo::renderScene()
    128 {
    129     std::string status = getStatus();
    130     float t = frame->updateWindowTitle(status.c_str());
    131     //std::cout << status << std::endl ; 
    132 
    133     updateUniform(t);
    134 
    135 #ifdef WITH_LOD
    136     cull->applyFork() ;
    137     cull->applyForkStreamQueryWorkaround() ;
    138     cull->dump("ICDemo::renderScene");
    139     //cull->pullback() ; 
    140 
    141     glUseProgram(draw->prog->program);
    142 
    143     for(unsigned lod=0 ; lod < num_lod ; lod++)
    144     {
    145         glBindVertexArray( use_cull ? this->drawVertexArray[lod] : this->allVertexArray);
    146 
    147         unsigned num_draw = use_cull ? clod->at(lod)->query_count : geom->num_inst ;
    148         if(num_draw == 0) continue ;
    149 
    150         const glm::uvec4& eidx = (*geom->eidx)[lod] ;
    151         glDrawElementsInstanced(GL_TRIANGLES, eidx.y, GL_UNSIGNED_INT, (void*)(eidx.x*sizeof(unsigned)), num_draw  ) ;
    152     }

    ///         element offset and num elements for each level are needed



DONE : Prim::Concatenate equivalent LODification in GMergedMesh::MakeLODComposite
-----------------------------------------------------------------------------------

::

    069 Prim* Prim::Concatenate( std::vector<Prim*> prims )
     70 {
     71     uint32_t ebufSize = 0;
     72     uint32_t vbufSize = 0;
     73 
     74     for(uint32_t p=0 ; p < prims.size() ; p++)
     75     {
     76         Prim* prim = prims[p];
     77         ebufSize += prim->ebuf->num_items ;
     78         vbufSize += prim->vbuf->num_items ;
     79     }
     80 
     81     uint32_t* edat =  new uint32_t[ebufSize] ;
     82     glm::vec4* vdat = new glm::vec4[vbufSize];
     83 
     84     Prim* concat = new Prim ;
     85 
     86     std::vector<glm::uvec4>& eidx = concat->eidx ;
     87     concat->ebuf = new Buf( ebufSize , sizeof(uint32_t)*ebufSize , edat );
     88     concat->vbuf = new Buf( vbufSize , sizeof(glm::vec4)*vbufSize , vdat );
     89 
     90     unsigned eOffset = 0;
     91     unsigned vOffset = 0;
     92 
     93     for(uint32_t p=0 ; p < prims.size() ; p++)
     94     {
     95         Prim* prim = prims[p];
     96         uint32_t num_elem = prim->ebuf->num_items ;
     97         uint32_t num_vert = prim->vbuf->num_items ;
     98 
     99         for (uint32_t e=0; e < num_elem ; e++) edat[eOffset+e] = *((uint32_t*)prim->ebuf->ptr + e) + vOffset ;
    100 
    101         eidx.push_back( {  eOffset, num_elem, vOffset, num_vert } );
    102 
    103         memcpy( (void*)( vdat + vOffset ), prim->vbuf->ptr , prim->vbuf->num_bytes );
    104         eOffset += num_elem ;
    105         vOffset += num_vert ;
    106     }
    107 
    108     concat->bb = BB::FromBuf(concat->vbuf);
    109     concat->ce = concat->bb->get_center_extent();
    110 
    111     return concat ;
    112 }




Add Components to GMergedMesh, testing with GMergedMeshTest (--mm)
--------------------------------------------------------------------

::

    simon:ggeo blyth$ op --j1707 --mm --debugger
    === op-cmdline-binary-match : finds 1st argument with associated binary : --mm
    ubin /usr/local/opticks/lib/GMergedMeshTest cfm --mm cmdline --j1707 --mm --debugger
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/GMergedMeshTest
    264 -rwxr-xr-x  1 blyth  staff  133956 Aug 31 19:39 /usr/local/opticks/lib/GMergedMeshTest
    proceeding.. : lldb /usr/local/opticks/lib/GMergedMeshTest -- --j1707 --mm --debugger
    (lldb) target create "/usr/local/opticks/lib/GMergedMeshTest"
    Current executable set to '/usr/local/opticks/lib/GMergedMeshTest' (x86_64).
    (lldb) settings set -- target.run-args  "--j1707" "--mm" "--debugger"
    (lldb) r
    Process 10573 launched: '/usr/local/opticks/lib/GMergedMeshTest' (x86_64)
    2017-08-31 19:39:40.142 INFO  [2117533] [GMergedMesh::dumpSolids@683] GMergedMesh::MakeComposite ce0 gfloat4      0.002      0.001    -17.937     57.939 

    ...

    0 ni[nf/nv/nidx/pidx] (528,266,107408, 11)  id[nidx,midx,bidx,sidx]  (107408, 20, 15,  0) 
    1 ni[nf/nv/nidx/pidx] (432,218,107409,107408)  id[nidx,midx,bidx,sidx]  (107409, 18, 16,  0) 
    2 ni[nf/nv/nidx/pidx] (240,122,107410,107409)  id[nidx,midx,bidx,sidx]  (107410, 16, 20,  0) 
    3 ni[nf/nv/nidx/pidx] (288,146,107411,107409)  id[nidx,midx,bidx,sidx]  (107411, 17, 21,  0) 
    4 ni[nf/nv/nidx/pidx] ( 96, 50,107412,107408)  id[nidx,midx,bidx,sidx]  (107412, 19, 13,  0) 
    5 ni[nf/nv/nidx/pidx] (528,266,107408, 11)  id[nidx,midx,bidx,sidx]  (107408, 20, 15,  0) 
    6 ni[nf/nv/nidx/pidx] (432,218,107409,107408)  id[nidx,midx,bidx,sidx]  (107409, 18, 16,  0) 
    7 ni[nf/nv/nidx/pidx] (240,122,107410,107409)  id[nidx,midx,bidx,sidx]  (107410, 16, 20,  0) 
    8 ni[nf/nv/nidx/pidx] (288,146,107411,107409)  id[nidx,midx,bidx,sidx]  (107411, 17, 21,  0) 
    9 ni[nf/nv/nidx/pidx] ( 96, 50,107412,107408)  id[nidx,midx,bidx,sidx]  (107412, 19, 13,  0) 

    2017-08-31 19:39:40.143 INFO  [2117533] [GMesh::dumpComponents@1029] test_GMergedMesh_MakeComposite.dumpComponents numComponents 2
       0      0    1584       0     802
       1   1584    1584     802     802



How to test the LOD ? Need option to switch on LOD creation/render for use from tboolean-
----------------------------------------------------------------------------------------------------

::

    tboolean-;tboolean-torus --lod 1 --lodconfig "levels=3,verbosity=2" --debugger 

    ## psychedelic flickery mess for outer box, with the quad mesh 3rd level 
    ##  ... so the levels are getting there 

::

    2017-09-01 16:58:51.115 INFO  [2338535] [OpticksViz::uploadGeometry@251] Opticks time 0.0000,20.0000,20.0000,0.0000 space 0.0000,0.0000,0.0000,400.0000 wavelength 60.0000,820.0000,20.0000,760.0000
    2017-09-01 16:58:51.141 INFO  [2338535] [Renderer::upload@197] Renderer::upload m_num_lod 3 m_indices_count 11736
    2017-09-01 16:58:51.141 INFO  [2338535] [GMesh::dumpComponents@1073] Renderer::upload numComponents 3
       0      0    3896       0   11688
       1   3896      12   11688      24
       2   3908       4   11712       8
    2017-09-01 16:58:51.144 INFO  [2338535] [Renderer::upload@197] Renderer::upload m_num_lod 3 m_indices_count 11736
    2017-09-01 16:58:51.144 INFO  [2338535] [GMesh::dumpComponents@1073] Renderer::upload numComponents 3
       0      0    3896       0   11688
       1   3896      12   11688      24
       2   3908       4   11712       8
    2017-09-01 16:58:51.144 INFO  [2338535] [Opt


::

    335     glm::uvec4 eidx(m_cur_faces, nface, m_cur_vertices, nvert );


::

    In [1]: 11688+24+8
    Out[1]: 11720

    In [2]: 3896+12+4
    Out[2]: 3912

    In [3]: (3896+12+4)*3
    Out[3]: 11736




::

    147         unsigned num_draw = use_cull ? clod->at(lod)->query_count : geom->num_inst ;
    148         if(num_draw == 0) continue ;
    149 
    150         const glm::uvec4& eidx = (*geom->eidx)[lod] ;
    151         glDrawElementsInstanced(GL_TRIANGLES, eidx.y, GL_UNSIGNED_INT, (void*)(eidx.x*sizeof(unsigned)), num_draw  ) ;
    152     }
    153 




LOD checking with test geometry
-----------------------------------------


Unclear where to do the LODing... for now::


    078 void GGeoTest::modifyGeometry()
     79 {
     80     const char* csgpath = m_config->getCsgPath();
     81     bool analytic = m_config->getAnalytic();
     82 
     83     if(csgpath) assert(analytic == true);
     84 
     85     GMergedMesh* tmm_ = create();
     86 
     87     GMergedMesh* tmm = m_lod > 0 ? GMergedMesh::MakeLODComposite(tmm_, m_lodconfig->levels ) : tmm_ ;
     88 
     89 
     90     char geocode =  analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED ;  // message to OGeo
     91     tmm->setGeoCode( geocode );
     92 
     93     if(tmm->isTriangulated())
     94     {
     95         tmm->setITransformsBuffer(NULL); // avoiding FaceRepeated complications 
     96     }
     97 
     98     //tmm->dump("GGeoTest::modifyGeometry tmm ");
     99     m_geolib->clear();
    100     m_geolib->setMergedMesh( 0, tmm );
    101 }




Which gets invoked::

    265 void OpticksGeometry::modifyGeometry()
    266 {
    267     assert(m_ok->hasOpt("test"));
    268     LOG(debug) << "OpticksGeometry::modifyGeometry" ;
    269 
    270     std::string testconf = m_fcfg->getTestConfig();
    271     
    272     m_ggeo->modifyGeometry( testconf.empty() ? NULL : testconf.c_str() );
    273 
    274     
    275     if(m_ggeo->getMeshVerbosity() > 2)
    276     {   
    277         GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);
    278         if(mesh0)
    279         {   
    280             mesh0->dumpSolids("OpticksGeometry::modifyGeometry mesh0");
    281             mesh0->save("$TMP", "GMergedMesh", "modifyGeometry") ;
    282         }
    283     }
    284 
    285     
    286     TIMER("modifyGeometry");
    287 }



     809 void GGeo::modifyGeometry(const char* config)
     810 {
     811     // NB only invoked with test option : "ggv --test" 
     812     //   controlled from OpticksGeometry::loadGeometry 
     813 
     814     GGeoTestConfig* gtc = new GGeoTestConfig(config);
     815 
     816     LOG(trace) << "GGeo::modifyGeometry"
     817               << " config [" << ( config ? config : "" ) << "]" ;
     818 
     819     assert(m_geotest == NULL);
     820 
     821     m_geotest = new GGeoTest(m_ok, gtc, this);
     822     m_geotest->modifyGeometry();
     823 
     824 }


    098 GMergedMesh* GGeoTest::create()
     99 {
    100     //TODO: unify all these modes into CSG 
    101     //      whilst still supporting the old partlist approach 
    102 
    103     const char* csgpath = m_config->getCsgPath();
    104     const char* mode = m_config->getMode();
    105 
    106     GMergedMesh* tmm = NULL ;
    107 
    108     if( mode != NULL && strcmp(mode, "PmtInBox") == 0)
    109     {
    110         tmm = createPmtInBox();
    111     }
    112     else
    113     {
    114         std::vector<GSolid*> solids ;
    115         if(csgpath != NULL)
    116         {
    117             assert( strlen(csgpath) > 3 && "unreasonable csgpath strlen");
    118             loadCSG(csgpath, solids);
    119         }
    120         else
    121         {
    122             unsigned int nelem = m_config->getNumElements();
    123             assert(nelem > 0);
    124             if(     strcmp(mode, "BoxInBox") == 0) createBoxInBox(solids);
    125             else  LOG(warning) << "GGeoTest::create mode not recognized " << mode ;
    126         }
    127         tmm = combineSolids(solids);
    128     }
    129     assert(tmm);
    130     return tmm ;
    131 }


    327 GMergedMesh* GGeoTest::combineSolids(std::vector<GSolid*>& solids)
    328 {
    329     unsigned verbosity = 3 ;
    330     GMergedMesh* tri = GMergedMesh::combine( 0, NULL, solids, verbosity );
    331 
    332     unsigned nelem = solids.size() ;
    333     GTransforms* txf = GTransforms::make(nelem); // identities
    334     GIds*        aii = GIds::make(nelem);        // placeholder (n,4) of zeros
    335 
    336     tri->setAnalyticInstancedIdentityBuffer(aii->getBuffer());
    337     tri->setITransformsBuffer(txf->getBuffer());
    338 
    339     //  OGeo::makeAnalyticGeometry  requires AII and IT buffers to have same item counts
    340 
    341     if(m_opticks->hasOpt("dbganalytic"))
    342     {
    343         GParts* pts = tri->getParts();
    344         pts->setName(m_config->getName());
    345         const char* msg = "GGeoTest::combineSolids --dbganalytic" ;
    346         pts->Summary(msg);
    347         pts->dumpPrimInfo(msg); // this usually dumps nothing as solid buffer not yet created
    348     }
    349     // collected pts are converted into primitives in GParts::makePrimBuffer
    350     return tri ;
    351 }




LOD/Cull forking 
----------------------

How to proceed:

* tidy VAO usage, for easy switching between the LODed transforms buffers 

* basis buffers too "evolved", use simple buffer with OpenGL capabilities
  similar to instcull- Buf ?

* Renderer treats buffers as transients just passing thru, 
  would be simpler to follow the instcull first class citizen buffers approach, 
  and give then OpenGL skills


* changing upload_GBuffer and upload_NPY to return a Buf holding vitals
  probably sufficient



icdemo uses a Buf4 to manage the forked instance transform buffers::


     68 void ICDemo::init()
     69 {
     70     geom->vbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
     71     geom->ebuf->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
     72     geom->ibuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
     73 
     74 #ifdef WITH_LOD
     75     // clod houses multiple buffers to grab the LOD forked instance transforms
     76     clod->x = geom->ibuf->cloneZero(); // CPU allocates and fills with zeros
     77     clod->y = geom->ibuf->cloneZero();
     78     clod->z = geom->ibuf->cloneZero();
     79 
     80     clod->x->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);  // GPU allocates only, no copying 
     81     clod->y->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
     82     clod->z->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
     83 
     84     //clod->devnull = new Buf(0,0,NULL);  // suspect zero-sized buffer is handled different, so use 1-byte buffer
     85     clod->devnull = new Buf(0,1,NULL);
     86     clod->devnull->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);  // zero sized buffer used with workaround
     87 
     88     
     89     cull->setupFork(geom->ibuf, clod) ;
     90 


::

    327 void Renderer::upload_buffers(NSlice* islice, NSlice* fslice)
    328 {
    ...
    371     NPY<float>* ibuf_orig = m_drawable->getITransformsBuffer();
    372     NPY<float>* ibuf = ibuf_orig ;
    373     setHasTransforms(ibuf != NULL);
    374 
    375     if(islice)
    376     {
    377         LOG(warning) << "Renderer::upload_buffers instance slicing ibuf with " << islice->description() ;
    378         ibuf = ibuf_orig->make_slice(islice);
    379     }
    ...
    386     if(m_instanced) assert(hasTransforms()) ;
    ...
    398     if(hasTransforms())
    399     {
    400         m_transforms = upload_NPY(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  ibuf, "transforms");
    401         m_itransform_count = ibuf->getNumItems() ;
    402     }

    ///  buffer id also stored inside ibuf 

::

    229  void NPYBase::setBufferId(int buffer_id)
    230 {
    231     m_buffer_id = buffer_id  ;
    232 }
    233  int NPYBase::getBufferId() const
    234 {
    235     return m_buffer_id ;
    236 }


::

    154 GLuint Renderer::upload_NPY(GLenum target, GLenum usage, NPY<float>* buf, const char* name)
    155 {
    156     BBufSpec* spec = buf->getBufSpec();
    157 
    158     GLuint id = upload(target, usage, spec, name );
    159 
    160     buf->setBufferId(id);
    161     buf->setBufferTarget(target);
    162 
    163     LOG(trace) << "Renderer::upload_NPY    "
    164               << std::setw(20) << name
    165               << " id " << std::setw(4) << id
    166               << " bytes " << std::setw(10) << spec->num_bytes
    167               ;
    168 
    169     return id ;
    170 }




