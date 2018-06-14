Geometry Review
==================

Issues
--------

* going via .dae and .gdml files is historical too : predating the OpticksCSG 
  .gltf analytic approach which was grafted on 


See also 
---------

* :doc:`scene` early thoughts on analytic geometry 


Direct from live G4 to geocache/gltf : how difficult ? 
-------------------------------------------------------- 

* :doc:`direct_to_gltf_feasibility`


Control Layers for geometry loading
--------------------------------------

okc.Opticks
      * commandline control, resource management
      * currently OpticksResource/BOpticksResource requires a geometry cache 
        which makes no sense in general, ie prior to exporting geometry 
        ... need to split off the code that makes this assumption 
        into a separate class "OpticksDetectorResource" ?

okg.OpticksHub   
      very high level steering

okg.OpticksGeometry   
      middle management

ggeo.GGeo
      worker


Worker Classes
----------------

AssimpGGeo
    importer that traverses the assimp (COLLADA) G4DAE geometry 
    populating GGeo as it goes 

    static int AssimpGGeo::load(GGeo* ggeo)  // GLoaderImpFunctionPtr


GBndLib
GMaterialLib
GSurfaceLib
GScintillatorLib
GSourceLib
    All above are GPropLib subclass constituents of GGeo     

GGeoLib 
    holder of GMergedMesh 

GGeoBase
    protocol pure virtual base guaranteeing that subclasses 
    provide accessors to the libs

GScene(GGeoBase)
    somewhat expediently GScene is held by GGeo 
    (cannot fly analytic only yet)

GGeo(GGeoBase)
    central structure holding libs of geomety objects, mesh-centric 

Nd
    mimimalistic structural nodes from the glTF,
    most use via NScene, populated from the glTF by NScene::import_r

    The "gltf" here is that written by sc.py:gdml2gltf : no effort was 
    made to make it work with "standard" gltf renderers, the gltf json 
    was just used as a convienent way to pass structure from the python 
    gdml parser into Opticks C++ (NScene).


NCSG
    coordinator for NPY arrays of small numbers of nodes, transforms, planes for 
    CSG solid shapes. Must be small as uses complete binary tree serialization.

    Created in python from the source GDML, csg.py does tree manipulations 
    to avoid deep trees.

    analytic/sc.py (structure)
    analytic/csg.py (solids)

    NCSG::LoadTree loads from nodes.npy transforms.npy etc.. creating a tree of nnode 
    instances 


NGLTF
    loads the gltf file written by bin/gdml2gltf.py 
    and provides easy interface to its content : transforms etc.. 
    as well as the underlying ygltf tree (YoctoGL)
    
    Currently revolves around loading from file. 

    See examples/UseYoctoGL/UseYoctoGL_Write.cc for a brief look
    at C++ construction of gltf structure.


NScene(NGLTF)

    * Does far too much for one class
    * NGLTF base class
    * loads NCSG extras
    * finds repeat geometry for instancing 
    * constructs Nd node tree from the gltf 
    * stores index keyed maps of Nd and NCSG 

GScene
    constituent m_scene NScene loads gltf with NScene::Load

GParts

    * constructed from NCSG instances 
    * is merged along with meshes to create combo GParts held by GMergedMesh  
    * provides analytic buffer interface consumed by OXRAP which copies to GPU 
   

Direct G4 to Opticks geo structure 
--------------------------------------

Legacy via file approach::

    LiveG4 -> GDMLfile -> bin/gdml2gltf.py -> GLTFfile -> NGLTF -> NScene nd tree -> GScene 

Where to jump to in the direct from G4 approach ? 

* complexity of NScene makes it not an apealing target for direct from G4,
  how is it used from GScene (ie what are the essentials that are needed)   

* NGLTF is tied to gltf structure, NScene is too to a lesser degree

* is there any benefit in going from G4 into Opticks via the GLTF memory structure ? NO

* GLTF is a transmission format, it aint a structure thats particularly easy to 
  use (my Nd tree is much easier) 

* GLTF is useful as a way to make the Opticks geocache format 
  renderable by GLTF standard supporting (OpenGL) renderers, 
  so this means are inverting the flow of GLTF (it needs to 
  becomes the output format of NScene rather than its input) 


How to proceed ? IDEA 1 : pull NdTree out of NScene, and populate it from there or X4Scene 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* NScene inherits from NGLTF could make that a constituent not inherited, 

  * that would be good : BUT too much work for the benefit : so NO

  * ACTUALLY DID THIS : IT WAS NOT TOO DIFFICULT, AND MAKES
    THINGS MUCH CLEARER 

    * CAN NOW VIEW THE m_ngltf instance as a nd/NCSG structure source 
      THAT CAN BE SWAPPED FOR AN ALTERNATIVE ONE FROM X4 
 
  
* create a stripped down NSceneBase to hold just the essential
  model (nd node tree) without the GLTF mechanics
  that gets populated from NScene 

  * hmm better, but thats using inheritance : so NO

* instead pull the structure out of NScene : ie all that is common 
  between GLTF and G4Live routes : namely the nd tree as its associations
  to nnode/NCSG and material  

  * called one of : NStructure/NKernel/NVolumeTree/*NdTree* 

  * *NdTree* must be transport agnostic, ie no dependency on 
    gltf or G4 : just a substrate to hold the structure and coordinate
    that can be populated in different ways:

    1. from gltf with NScene
    2. from G4Live with new class X4Scene holding X4PhysicalVolume


* alternative would be to move the gltf mechanics out from NScene : but 
  thats much harder that creating new ... can rename classes if necessary 
  once the rejig is done (as the NScene name was originally intended be the "NdTree" 
  but it got swapped with gltf mechanics) 
  

How to proceed ? IDEA 2 : direct to GScene/GNode/GSolid ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

* to see if this is plausible need to see how does GScene use NScene ? 

* Could GScene be moved over to consuming NdTree ?

* would like to get rid of GScene eventually : it steals many constituents 
  of GGeo anyhow  : but that is not-adiabatic enough 


IDEA 3 
~~~~~~~~~

Start by tidying up NScene:

* distancing NScene from NGLTF, so NScene can work with an alternative source of nodes/meshes too 
* make NGLTF a constituent
* move gltf mechanics from NScene into NGLTF
* provide NGLTF with a higher level interface,
  ie that hides specifics of gltf transport 


FURTHER CLEANUP
~~~~~~~~~~~~~~~~~~

* review all use of m_ngltf within NScene, aiming to make 
  a higher level interface by moving specifics into NGLTF
  and the to be created X4Scene? 

  For example with a gltf source the NCSG are loaded from gltf extras  
  but with X4 the NCSG will be constructed directly from translated nnode, 
  thus NScene needs to be above these details, moving that and other 
  handling into the sources such that they can present a common interface 
  to NScene.


* instanciate NGLTF by loading from the file outside NScene, and pass
  to NScene as ctor argument to be stored in m_source (replacing m_ngltf)





Exercise the old route to check have not broken humpty
----------------------------------------------------------


::

    op --gdml2gltf 
 
    cp /usr/local/opticks-cmake-overhaul/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf /tmp/

    cp -r /usr/local/opticks-cmake-overhaul/opticksdata/export/DayaBay_VGDX_20140414-1300/extras /tmp/
    ## the gltf refers to lots of extras, which have to travel with it 

    opticks-pretty /tmp/g4_00.gltf



GParts
---------

Single Tree GParts created from from NCSG by GScene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GParts are created from the NCSG in GScene::createVolume where they get attached to a GSolid::

    629 GSolid* GScene::createVolume(nd* n, unsigned depth, bool& recursive_select  ) // compare with AssimpGGeo::convertStructureVisit
    630 {
    ...
    644     NCSG*   csg =  getCSG(rel_mesh_idx);

    661     std::string bndspec = lookupBoundarySpec(solid, n);  // using just transferred boundary from tri branch
    662 
    663     GParts* pts = GParts::make( csg, bndspec.c_str(), m_verbosity  ); // amplification from mesh level to node level 
    664 
    665     pts->setBndLib(m_tri_bndlib);
    666 
    667     solid->setParts( pts );



GScene
--------

::

     585 GSolid* GScene::createVolumeTree(NScene* scene) // creates analytic GSolid/GNode tree without access to triangulated GGeo info
     586 {       
     587     if(m_verbosity > 0)
     588     LOG(info) << "GScene::createVolumeTree START"
     589               << "  verbosity " << m_verbosity
     590               << " query " << m_query->description()
     591               ;
     592     assert(scene);
     593 
     594     //scene->dumpNdTree("GScene::createVolumeTree");
     595         
     596     nd* root_nd = scene->getRoot() ;
     597     assert(root_nd->idx == 0 );
     598         
     599     GSolid* parent = NULL ;
     600     unsigned depth = 0 ; 
     601     bool recursive_select = false ; 
     602     GSolid* root = createVolumeTree_r( root_nd, parent, depth, recursive_select );
     603     assert(root);
     604 
     605     assert( m_nodes.size() == scene->getNumNd()) ;
     606         
     607     if(m_verbosity > 0)
     608     LOG(info) << "GScene::createVolumeTree DONE num_nodes: " << m_nodes.size()  ;
     609     return root ; 
     610 }              


NCSG : serialization ctor boost from nnode tree
-------------------------------------------------

::

     088 // ctor : booting from in memory node tree
      89 NCSG::NCSG(nnode* root )
      90    :
      91    m_meta(NULL),
      92    m_treedir(NULL),
      93    m_index(0),
      94    m_surface_epsilon(SURFACE_EPSILON),
      95    m_verbosity(root->verbosity),
      96    m_usedglobally(false),
      97    m_root(root),
      98    m_points(NULL),
      99    m_uncoincide(make_uncoincide()),
     100    m_nudger(make_nudger()),
     101    m_nodes(NULL),
     102    m_transforms(NULL),
     103    m_gtransforms(NULL),
     104    m_planes(NULL),
     105    m_srcverts(NULL),
     106    m_srcfaces(NULL),
     107    m_num_nodes(0),
     108    m_num_transforms(0),
     109    m_num_planes(0),
     110    m_num_srcverts(0),
     111    m_num_srcfaces(0),
     112    m_height(root->maxdepth()),
     113    m_boundary(NULL),
     114    m_config(NULL),
     115    m_gpuoffset(0,0,0),
     116    m_container(0),
     117    m_containerscale(2.f),
     118    m_tris(NULL)
     119 {
     120 
     121    setBoundary( root->boundary );
     122 
     123    m_num_nodes = NumNodes(m_height);
     124 
     125    m_nodes = NPY<float>::make( m_num_nodes, NJ, NK);
     126    m_nodes->zero();
     127 
     128    m_transforms = NPY<float>::make(0,NTRAN,4,4) ;
     129    m_transforms->zero();
     130 
     131    m_gtransforms = NPY<float>::make(0,NTRAN,4,4) ;
     132    m_gtransforms->zero();
     133 
     134    m_planes = NPY<float>::make(0,4);
     135    m_planes->zero();
     136 
     137    m_meta = new NParameters ;
     138 }




G4Hype vs Opticks CSG_HYPERBOLOID : can I relate them ?
----------------------------------------------------------


::

    071   G4Hype(const G4String& pName,
     72                G4double  newInnerRadius,
     73                G4double  newOuterRadius,
     74                G4double  newInnerStereo,
     75                G4double  newOuterStereo,
     76                G4double  newHalfLenZ);
       

::

    127 inline
    128 G4double G4Hype::HypeInnerRadius2(G4double zVal) const
    129   {
    130     return (tanInnerStereo2*zVal*zVal+innerRadius2);
    131   } 
    ///
    ///         x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )
    ///                     =  r0^2 +  (r0/zf)^2 * z^2
    ///
    ///           tanStereo = r0/zf
    ///
    ///       -->  zf = r0/tanStereo
    ///
    ///        newHalfLenZ -> 
    ///
    ///
    132 
    133 inline
    134 G4double G4Hype::HypeOuterRadius2(G4double zVal) const
    135   {
    136     return (tanOuterStereo2*zVal*zVal+outerRadius2);
    137   }




::

     560 static __device__
     561 bool csg_intersect_hyperboloid(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
     562 {
     563    /*
     564      http://mathworld.wolfram.com/One-SheetedHyperboloid.html
     565 
     566       x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )
     567       x^2 + y^2 - (r0^2/zf^2) * z^2 - r0^2  =  0 
     568       x^2 + y^2 + A * z^2 + B   =  0 
     569    
     570       grad( x^2 + y^2 + A * z^2 + B ) =  [2 x, 2 y, A*2z ] 
     571 
     572  
     573      (ox+t sx)^2 + (oy + t sy)^2 + A (oz+ t sz)^2 + B = 0 
     574 
     575       t^2 ( sxsx + sysy + A szsz ) + 2*t ( oxsx + oysy + A * ozsz ) +  (oxox + oyoy + A * ozoz + B ) = 0 
     576 
     577    */
     578 
     579     const float zero(0.f);
     580     const float one(1.f);
     581 
     582     const float r0 = q0.f.x ;  // waist (z=0) radius 
     583     const float zf = q0.f.y ;  // at z=zf radius grows to  sqrt(2)*r0 
     584     const float z1 = q0.f.z ;  // z1 < z2 by assertion  
     585     const float z2 = q0.f.w ;
     586 
     587     const float rr0 = r0*r0 ;
     588     const float z1s = z1/zf ;
     589     const float z2s = z2/zf ;
     590     const float rr1 = rr0 * ( z1s*z1s + one ) ; // radii squared at z=z1, z=z2
     591     const float rr2 = rr0 * ( z2s*z2s + one ) ;
     592 
     593     const float A = -rr0/(zf*zf) ;
     594     const float B = -rr0 ;
     595 






G4GDML Writing Solids
-----------------------

G4GDMLWriteStructure::TraverseVolumeTree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Primary AddSolid invokation happens at the end of the recursive tail of 
the structure traverse::

    381 
    382 G4Transform3D G4GDMLWriteStructure::
    383 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    384 {
    ...
    539    structureElement->appendChild(volumeElement);
    540    // Append the volume AFTER traversing the children so that
    541    // the order of volumes will be correct!
    542 
    543    VolumeMap()[tmplv] = R;
    544 
    545    AddExtension(volumeElement, volumePtr);
    546    // Add any possible user defined extension attached to a volume
    547 
    548    AddMaterial(volumePtr->GetMaterial());
    549    // Add the involved materials and solids!
    550 
    551    AddSolid(solidPtr);
    552 
    553    SkinSurfaceCache(GetSkinSurface(volumePtr));
    554 
    555    return R;
    556 }


G4GDMLWriteSolids::SolidsWrite G4GDMLWriteStructure::StructureWrite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setup the "child-of-root" level solids element and clear the list of instances::

    1022 void G4GDMLWriteSolids::SolidsWrite(xercesc::DOMElement* gdmlElement)
    1023 {
    1024    G4cout << "G4GDML: Writing solids..." << G4endl;
    1025 
    1026    solidsElement = NewElement("solids");
    1027    gdmlElement->appendChild(solidsElement);
    1028 
    1029    solidList.clear();
    1030 }
    1031 

The "structure" element is also "child-of-root":: 


    374 void G4GDMLWriteStructure::StructureWrite(xercesc::DOMElement* gdmlElement)
    375 {
    376    G4cout << "G4GDML: Writing structure..." << G4endl;
    377 
    378    structureElement = NewElement("structure");
    379    gdmlElement->appendChild(structureElement);
    380 }







G4GDMLWriteSolids::AddSolid(G4VSolid* ) subclass fanout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* skip G4VSolid instances that have been added already

* dynamic_cast to identify subclass, then call Write method
  specific to the subclass

  * many of the Write methods (for composites/booleans) 
    will first invoke AddSolid for their constituents before
    writing the elements for themselves using name references 
    to constituents


::


    1032 void G4GDMLWriteSolids::AddSolid(const G4VSolid* const solidPtr)
    1033 {
    1034    for (size_t i=0; i<solidList.size(); i++)   // Check if solid is
    1035    {                                           // already in the list!
    1036       if (solidList[i] == solidPtr)  { return; }
    1037    }
    1038 
    1039    solidList.push_back(solidPtr);
    1040 
    1041    if (const G4BooleanSolid* const booleanPtr
    1042      = dynamic_cast<const G4BooleanSolid*>(solidPtr))
    1043      { BooleanWrite(solidsElement,booleanPtr); } else
    1044    if (solidPtr->GetEntityType()=="G4MultiUnion")
    1045      { const G4MultiUnion* const munionPtr
    1046      = static_cast<const G4MultiUnion*>(solidPtr);





Analytic GScene uses the GGeo proplibs for material/surface props...
------------------------------------------------------------------------

* unified analytic-triangulated gltf geometry would need to include all these

::

      46       
      47 // for some libs there is no analytic variant 
      48 GMaterialLib*     GScene::getMaterialLib() {     return m_ggeo->getMaterialLib(); }
      49 GSurfaceLib*      GScene::getSurfaceLib() {      return m_ggeo->getSurfaceLib(); }
      50 GBndLib*          GScene::getBndLib() {          return m_ggeo->getBndLib(); }
      51 GPmtLib*          GScene::getPmtLib() {          return m_ggeo->getPmtLib(); }
      52 GScintillatorLib* GScene::getScintillatorLib() { return m_ggeo->getScintillatorLib(); }
      53 GSourceLib*       GScene::getSourceLib() {       return m_ggeo->getSourceLib(); }
      54 



Geometry consumers : what is actually needed ?
------------------------------------------------

oxrap.OGeo


oxrap.OScene
--------------

Canonical m_scene instance resides in okop-/OpEngine 

OScene::init creates the OptiX context and populates
it with geometry, boundary etc.. info 



oxrap.OGeo : operates from analytic or triangulated 
----------------------------------------------------------

* GParts associated with each GMergedMesh hold the analytic geometry

::

     614 optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm, unsigned lod)
     615 {
     616     if(m_verbosity > 2)
     617     LOG(warning) << "OGeo::makeAnalyticGeometry START"
     618                  << " verbosity " << m_verbosity
     619                  << " lod " << lod
     620                  << " mm " << mm->getIndex()
     621                  ;
     622 
     623     // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest
     624 
     625     GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry ");
     626 
     627 





Questions
------------

* How difficult to create NScene direct from live G4 ?



High Level G4DAE COLLADA Writing
-----------------------------------

/usr/local/opticks-cmake-overhaul/externals/g4dae/g4dae-opticks/src/G4DAEWrite.cc::

    179 G4Transform3D G4DAEWrite::Write(const G4String& fname,
    180                                  const G4LogicalVolume* const logvol,
    181                                  const G4String& setSchemaLocation,
    182                                  const G4int depth,
    183                                        G4bool refs,
    184                                        G4bool _recreatePoly,
    185                                        G4int nodeIndex )
    186 {
    ...
    212    doc = impl->createDocument(0,tempStr,0);
    213    xercesc::DOMElement* dae = doc->getDocumentElement();
    214 
    ...
    233    dae->setAttributeNode(NewAttribute("xmlns",
    234                           "http://www.collada.org/2005/11/COLLADASchema"));
    235    dae->setAttributeNode(NewAttribute("version","1.4.1"));
    ...
    243    AssetWrite(dae);
    244    EffectsWrite(dae);
    245    SolidsWrite(dae);   // geometry before materials to match pycollada
    ///
    ///    SolidsWrite just opens the library_geometry element ... actual writing 
    ///    of solids done in recursive tail of TraverseVolumeTree  by G4DAEWriteSolids::AddSolid

    246    MaterialsWrite(dae);
    ///    ditto ... G4DAEWriteMaterials::AddMaterial
    ///
    247 
    248    StructureWrite(dae);   // writing order does not follow inheritance order
    249 
    250    SetupWrite(dae, logvol);
    251 
    252    G4Transform3D R = TraverseVolumeTree(logvol,depth);
    253 
    254    SurfacesWrite();
    255 
    256    xercesc::XMLFormatTarget *myFormTarget =
    257      new xercesc::LocalFileFormatTarget(fname.c_str());
    258 
    259    try
    260    {
    261 #if XERCES_VERSION_MAJOR >= 3
    262                                             // DOM L3 as per Xerces 3.0 API
    263       xercesc::DOMLSOutput *theOutput =
    264         ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
    265       theOutput->setByteStream(myFormTarget);
    266       writer->write(doc, theOutput);
    267 #else
    268       writer->writeNode(myFormTarget, *doc);
    269 #endif


* note that BorderSurface are collected within TraverseVolumeTree




NScene(NGLTF)
----------------

Used by GGeo::loadFromGLTF and GScene, GGeo.cc::

     658     m_nscene = new NScene(gltfbase, gltfname, gltfconfig);
     659     m_gscene = new GScene(this, m_nscene );

Scene files in glTF format are created by opticks/analytic/sc.py 
which parses the input GDML geometry file and writes the mesh (ie solid 
shapes) as np ncsg and the tree structure as json/gltf.

NScene imports the gltf using its NGLTF based (YoctoGL external)
creating a nd tree. The small CSG node trees for each solid
are polygonized on load in NScene::load_mesh_extras.

* somehere the Geant4 polygonizations are swapped in 


opticksgeo.OpticksHub (okg-)
-----------------------------

Starts out with most things NULL, populated in init::

    138 OpticksHub::OpticksHub(Opticks* ok)
    139    :
    140    m_log(new SLog("OpticksHub::OpticksHub")),
    141    m_ok(ok),
    142    m_gltf(-1),        // m_ok not yet configured, so defer getting the settings
    143    m_run(m_ok->getRun()),
    144    m_geometry(NULL),
    145    m_ggeo(NULL),
    146    m_gscene(NULL),
    147    m_composition(new Composition),
    148 #ifdef OPTICKS_NPYSERVER
    149    m_delegate(NULL),
    150    m_server(NULL)
    151 #endif
    152    m_cfg(new BCfg("umbrella", false)),
    153    m_fcfg(m_ok->getCfg()),
    154    m_state(NULL),
    155    m_lookup(new NLookup()),
    156    m_bookmarks(NULL),
    157    m_gen(NULL),
    158    m_gun(NULL),
    159    m_aim(NULL),
    160    m_geotest(NULL),
    161    m_err(0)
    162 {
    163    init();
    164    (*m_log)("DONE");
    165 }

    167 void OpticksHub::init()
    168 {
    169     add(m_fcfg);
    170 
    171     configure();
    172     configureServer();
    173     configureCompositionSize();
    174     configureLookupA();
    175 
    176     m_aim = new OpticksAim(this) ;
    177 
    178     loadGeometry() ;
    179     if(m_err) return ;
    180 
    181     configureGeometry() ;
    182 
    183     m_gen = new OpticksGen(this) ;
    184     m_gun = new OpticksGun(this) ;
    185 }

    208 void OpticksHub::configure()
    209 {   
    210     m_composition->addConfig(m_cfg);
    211     //m_cfg->dumpTree();
    212     
    213     int argc    = m_ok->getArgc();
    214     char** argv = m_ok->getArgv();
    215     
    216     LOG(debug) << "OpticksHub::configure " << argv[0] ;
    217     
    218     m_cfg->commandline(argc, argv);
    219     m_ok->configure();
    220     
    221     if(m_fcfg->hasError())
    222     {   
    223         LOG(fatal) << "OpticksHub::config parse error " << m_fcfg->getErrorMessage() ;
    224         m_fcfg->dump("OpticksHub::config m_fcfg");
    225         m_ok->setExit(true);
    226         return ;
    227     }
    228     
    229     m_gltf =  m_ok->getGLTF() ;
    230     LOG(info) << "OpticksHub::configure"
    231               << " m_gltf " << m_gltf
    232               ;
    233     
    234     bool compute = m_ok->isCompute();
    235     bool compute_opt = hasOpt("compute") ;
    236     if(compute && !compute_opt)
    237         LOG(warning) << "OpticksHub::configure FORCED COMPUTE MODE : as remote session detected " ;
    238     
    239     
    240     if(hasOpt("idpath")) std::cout << m_ok->getIdPath() << std::endl ;
    241     if(hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
    242     if(hasOpt("help|version|idpath"))
    243     {   
    244         m_ok->setExit(true);
    245         return ;
    246     }
    247     
    248     
    249     if(!m_ok->isValid())
    250     {   
    251         // defer death til after getting help
    252         LOG(fatal) << "OpticksHub::configure OPTICKS INVALID : missing envvar or geometry path ?" ;
    253         assert(0);
    254     }
    255 }


     



okg-.OpticksHub::loadGeometry
-------------------------------

::

    356 void OpticksHub::loadGeometry()
    357 {   
    358     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
    359     
    360     LOG(info) << "OpticksHub::loadGeometry START" ;
    361     
    362     
    363     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
    364     
    365     m_geometry->loadGeometry();
    366     
    367     m_ggeo = m_geometry->getGGeo();
    368     
    369     m_gscene = m_ggeo->getScene();
    370     
    371     
    372     //   Lookup A and B are now set ...
    373     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    374     //      B : on GGeo loading in GGeo::setupLookup
    375     
    ...     skip test geometry handling 
    ...
    399     registerGeometry();
    400     
    401     
    402     m_ggeo->setComposition(m_composition);
    403     
    404     LOG(info) << "OpticksHub::loadGeometry DONE" ;
    405 }   



okg-.OpticksGeometry::loadGeometry
-----------------------------------

::

     77 void OpticksGeometry::init()
     78 {
     79     bool geocache = !m_fcfg->hasOpt("nogeocache") ;
     80     bool instanced = !m_fcfg->hasOpt("noinstanced") ; // find repeated geometry 
     81 
     82     LOG(debug) << "OpticksGeometry::init"
     83               << " geocache " << geocache
     84               << " instanced " << instanced
     85               ;
     86 
     87     m_ok->setGeocache(geocache);
     88     m_ok->setInstanced(instanced); // find repeated geometry 
     89 
     90     m_ggeo = new GGeo(m_ok);
     91     m_ggeo->setLookup(m_hub->getLookup());
     92 }
     93 


     117 // setLoaderImp : sets implementation that does the actual loading
     118 // using a function pointer to the implementation 
     119 // avoids ggeo-/GLoader depending on all the implementations
     120 
     121 void GGeo::setLoaderImp(GLoaderImpFunctionPtr imp)
     122 {   
     123     m_loader_imp = imp ;
     124 }


::

    132 void OpticksGeometry::loadGeometryBase()
    133 {
    134     LOG(error) << "OpticksGeometry::loadGeometryBase START " ;
    135     OpticksResource* resource = m_ok->getResource();
    136 
    137     if(m_ok->hasOpt("qe1"))
    138         m_ggeo->getSurfaceLib()->setFakeEfficiency(1.0);
    139 
    140 
    141     m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    142 
    143 
    144     m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);
    145     m_ggeo->setMeshVerbosity(m_fcfg->getMeshVerbosity());
    146     m_ggeo->setMeshJoinCfg( resource->getMeshfix() );
    147 
    148     std::string meshversion = m_fcfg->getMeshVersion() ;;
    149     if(!meshversion.empty())
    150     {
    151         LOG(warning) << "OpticksGeometry::loadGeometry using debug meshversion " << meshversion ;
    152         m_ggeo->getGeoLib()->setMeshVersion(meshversion.c_str());
    153     }
    154 
    155     m_ggeo->loadGeometry();   // potentially from cache : for gltf > 0 loads both tri and ana geometry 
    156 
    157     if(m_ggeo->getMeshVerbosity() > 2)
    158     {
    159         GMergedMesh* mesh1 = m_ggeo->getMergedMesh(1);
    160         if(mesh1)
    161         {
    162             mesh1->dumpSolids("OpticksGeometry::loadGeometryBase mesh1");
    163             mesh1->save("$TMP", "GMergedMesh", "baseGeometry") ;
    164         }
    165     }
    166 
    167     LOG(error) << "OpticksGeometry::loadGeometryBase DONE " ;
    168     TIMER("loadGeometryBase");
    169 }




When running precache GGeo::init creates the various libs in 
preparation to be populated during the traverse.::

     336 void GGeo::init()
     337 {
     338    const char* idpath = m_ok->getIdPath() ;
     339    LOG(trace) << "GGeo::init"
     340               << " idpath " << ( idpath ? idpath : "NULL" )
     341               ;  
     342               
     343    assert(idpath && "GGeo::init idpath is required" );
     344    
     345    fs::path geocache(idpath);
     346    bool cache_exists = fs::exists(geocache) && fs::is_directory(geocache) ;
     347    bool cache_requested = m_ok->isGeocache() ; 
     348    
     349    m_loaded = cache_exists && cache_requested ;
     350    
     351    LOG(trace) << "GGeo::init"
     352              << " idpath " << idpath
     353              << " cache_exists " << cache_exists
     354              << " cache_requested " << cache_requested
     355              << " m_loaded " << m_loaded 
     356              ;
     357              
     358    if(m_loaded) return ;
     359    
     360    //////////////  below only when operating pre-cache //////////////////////////
     361    
     362    m_bndlib = new GBndLib(m_ok);
     363    m_materiallib = new GMaterialLib(m_ok);
     364    m_surfacelib  = new GSurfaceLib(m_ok);
     365    
     366    m_bndlib->setMaterialLib(m_materiallib);
     367    m_bndlib->setSurfaceLib(m_surfacelib);
     368    
     369    // NB this m_analytic is always false
     370    //    the analytic versions of these libs are born in GScene
     371    assert( m_analytic == false );  
     372    bool testgeo = false ;  
     373    
     374    m_meshlib = new GMeshLib(m_ok, m_analytic);
     375    m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
     376    m_nodelib = new GNodeLib(m_ok, m_analytic, testgeo );
     377    
     378    m_treecheck = new GTreeCheck(m_geolib, m_nodelib, m_ok->getSceneConfig() ) ;
     379    
     380    
     381    GColorizer::Style_t style = GColorizer::PSYCHEDELIC_NODE ;
     382    OpticksColors* colors = getColors();
     383    
     384    m_colorizer = new GColorizer( m_nodelib, m_geolib, m_bndlib, colors, style ); // colorizer needs full tree, so pre-cache only 
     385 
     386 
     387    m_scintillatorlib  = new GScintillatorLib(m_ok);
     388    m_sourcelib  = new GSourceLib(m_ok);
     389 
     390    m_pmtlib = NULL ;
     391 
     392    LOG(trace) << "GGeo::init DONE" ;
     393 }



::

     503 void GGeo::loadGeometry()
     504 {
     505     bool loaded = isLoaded() ;
     506 
     507     int gltf = m_ok->getGLTF();
     508 
     509     LOG(info) << "GGeo::loadGeometry START"
     510               << " loaded " << loaded
     511               << " gltf " << gltf
     512               ;
     513 
     514     if(!loaded)
     515     {
     516         loadFromG4DAE();
     517         save();
     518 
     519         if(gltf > 0 && gltf < 10)
     520         {
     521             loadAnalyticFromGLTF();
     522             saveAnalytic();
     523         }
     524     }
     525     else
     526     {
     527         loadFromCache();
     528         if(gltf > 0 && gltf < 10)
     529         {
     530             loadAnalyticFromCache();
     531         }
     532     }
     533 
     534 
     535     if(m_ok->isAnalyticPMTLoad())
     536     {
     537         m_pmtlib = GPmtLib::load(m_ok, m_bndlib );
     538     }
     539 
     540     if( gltf >= 10 )
     541     {
     542         LOG(info) << "GGeo::loadGeometry DEBUGGING loadAnalyticFromGLTF " ;
     543         loadAnalyticFromGLTF();
     544     }
     545 
     546     setupLookup();
     547     setupColors();
     548     setupTyp();
     549     LOG(info) << "GGeo::loadGeometry DONE" ;
     550 }



The current standard loader in the assimp loader.  


::


     552 void GGeo::loadFromG4DAE()
     553 {
     554     LOG(error) << "GGeo::loadFromG4DAE START" ;
     555 
     556     int rc = (*m_loader_imp)(this);   //  imp set in OpticksGeometry::loadGeometryBase, m_ggeo->setLoaderImp(&AssimpGGeo::load); 
     557 
     558     if(rc != 0)
     559         LOG(fatal) << "GGeo::loadFromG4DAE"
     560                    << " FAILED : probably you need to download opticksdata "
     561                    ;
     562 
     563     assert(rc == 0 && "G4DAE geometry file does not exist, try : opticksdata- ; opticksdata-- ") ;
     564 
     565     prepareScintillatorLib();
     566 
     567     prepareMeshes();
     568 
     569     prepareVertexColors();
     570 
     571     LOG(error) << "GGeo::loadFromG4DAE DONE" ;
     572 }


