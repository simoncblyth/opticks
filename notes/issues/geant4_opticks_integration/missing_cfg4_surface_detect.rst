Missing CFG4 Surface Detect
==============================

Issue
-------

* OKG4Test running with DYB GDML geometry produces no hits, as no photons get flagged SURFACE_DETECT 
* BUT tpmt-- --okg4 running succeeds to produce hits and "SD" that matches between OK and G4   

Observations
-------------

* below code trace suggests lack of "SD" due to the GDML detector missing optical surfaces 
  (with EFFICIENCY values) on PMTs ? 


Fixed ? N
-----------------------------------------------------------------------

::

   OKG4Test --save

   lldb OKG4Test -- --compute --save

   (lldb) b "OpRayleigh::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)" 

       ## torch running : G4 thinks in vacuum.... 

   OKG4Test --g4gun --compute --save



CGDMLDetector vs CTestDetector
-------------------------------


Writing G4DAE Optical surfaces
---------------------------------

g4dae/src/G4DAEWriteStructure.cc::

    499 G4Transform3D G4DAEWriteStructure::
    500 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    501 {
    ...
    552    const G4int daughterCount = volumePtr->GetNoDaughters();
    ...
    557    for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
    558    {
    559       const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
    ...
    562       G4Transform3D daughterR;
    ...
    564       daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);
    565 
    566       G4RotationMatrix rot, invrot;
    567       if (physvol->GetFrameRotation() != 0)
    568       {
    569          rot = *(physvol->GetFrameRotation());
    570          invrot = rot.inverse();
    571       }
    572 
    573       // G4Transform3D P(rot,physvol->GetObjectTranslation());  GDML does this : not inverting the rotation portion 
    574       G4Transform3D P(invrot,physvol->GetObjectTranslation());
    575 
    576       PhysvolWrite(nodeElement,physvol,invR*P*daughterR,ModuleName);
    577       BorderSurfaceCache(GetBorderSurface(physvol));
    ///
    ///               GetBorderSurface   :  returns first G4LogicalBorderSurface* for which "pv1" == physvol 
    ///               BorderSurfaceCache :  forms and collects bordersurface and associated opticalsurface XML elements    
    ///
    578    }
    ...
    581    structureElement->appendChild(nodeElement);
    ...
    586    VolumeMap()[volumePtr] = R;  // always identity matrix ?
    ...
    592    SkinSurfaceCache(GetSkinSurface(volumePtr));
    ///
    ///               GetSkinSurface    :  returns first G4LogicalSkinSurface* associated with G4LogicalVolume*  volumePtr
    ///               SkinSurfaceCache  :  forms and collects skinsurface and associated opticalsurface XML elements  
    ///
    593 
    594    return R;
    595 }



Reading G4DAE/ggeo Optical Surfaces
---------------------------------------


assimprap/AssimpGGeo.cc::

     359 void AssimpGGeo::convertMaterials(const aiScene* scene, GGeo* gg, const char* query )
     360 {
     361     LOG(info)<<"AssimpGGeo::convertMaterials "
     362              << " query " << query
     363              << " mNumMaterials " << scene->mNumMaterials
     364              ;
     365 
     366     //GDomain<float>* standard_domain = gg->getBoundaryLib()->getStandardDomain(); 
     367     GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain();
     368 
     369 
     370     for(unsigned int i = 0; i < scene->mNumMaterials; i++)
     371     {
     372         unsigned int index = i ;  // hmm, make 1-based later 
     373 
     374         aiMaterial* mat = scene->mMaterials[i] ;
     375 
     376         aiString name_;
     377         mat->Get(AI_MATKEY_NAME, name_);
     378 
     379         const char* name = name_.C_Str();
     380 
     381         //if(strncmp(query, name, strlen(query))!=0) continue ;  
     382 
     383         LOG(debug) << "AssimpGGeo::convertMaterials " << i << " " << name ;
     384 
     385         const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
     386         const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );
     387 
     388         const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );
     389 
     390         const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
     391         const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
     392         const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
     393         const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
     394         const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );
     395 
     396 
     397         GOpticalSurface* os = osnam && ostyp && osmod && osfin && osval ? new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) : NULL ;
     ...
     413         if( sslv )
     414         {
     415             assert(os && "all ss must have associated os");
     416 
     417             GSkinSurface* gss = new GSkinSurface(name, index, os);
     418 
     419 
     420             LOG(debug) << "AssimpGGeo::convertMaterials GSkinSurface "
     421                       << " name " << name
     422                       << " sslv " << sslv
     423                       ;
     424 
     425             gss->setStandardDomain(standard_domain);
     426             gss->setSkinSurface(sslv);
     427             addProperties(gss, mat );
     428 
     429             LOG(debug) << gss->description();
     430             gg->add(gss);
     431 
     432             {
     433                 // without standard domain applied
     434                 GSkinSurface*  gss_raw = new GSkinSurface(name, index, os);
     435                 gss_raw->setSkinSurface(sslv);
     436                 addProperties(gss_raw, mat );
     437                 gg->addRaw(gss_raw);  // this was erroreously gss for a long time
     438             }  
     439 
     440         }
     441         else if (bspv1 && bspv2 )
     442         {
     443             assert(os && "all bs must have associated os");
     444             GBorderSurface* gbs = new GBorderSurface(name, index, os);
     445 
     446             gbs->setStandardDomain(standard_domain);
     447             gbs->setBorderSurface(bspv1, bspv2);
     448             addProperties(gbs, mat );
     449 
     450             LOG(debug) << gbs->description();
     451 
     452             gg->add(gbs);



* GSkinSurface and GBorderSurface holding the volume names are added to GGeo

::

     228 GSkinSurface* GGeo::getSkinSurface(unsigned int index)
     229 {
     230     return m_skin_surfaces[index];
     231 }
     232 GBorderSurface* GGeo::getBorderSurface(unsigned int index)
     233 {
     234     return m_border_surfaces[index];
     235 }


Hmm volume association not persisted in slib:: 

     167 void GSurfaceLib::add(GBorderSurface* raw)
     168 {
     169     GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
     170     add(surf);
     171 }
     172 void GSurfaceLib::add(GSkinSurface* raw)
     173 {
     174     LOG(trace) << "GSurfaceLib::add(GSkinSurface*) " << ( raw ? raw->getName() : "NULL" ) ;
     175     GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
     176     add(surf);
     177 }


GGeo associates imat/isur/osur/omat guint4 boundary index with GSolid(GNode).

::

    0832 GSolid* AssimpGGeo::convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* /*parent*/)
     833 {
     834     // Associates node to extra information analogous to collada_to_chroma.py:visit
     835     //

     908     GSolid* solid = new GSolid(nodeIndex, gtransform, mesh, UINT_MAX, NULL ); // sensor starts NULL
     909     solid->setLevelTransform(ltransform);
     910 
     911     const char* lv   = node->getName(0);
     912     const char* pv   = node->getName(1);
     913     const char* pv_p   = pnode->getName(1);
     914 
     915     gg->countMeshUsage(msi, nodeIndex, lv, pv);
     916 
     917     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
     918     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
     919     GSkinSurface*   sks = gg->findSkinSurface(lv);
     ...
     991     GBndLib* blib = gg->getBndLib();
     992     GSurfaceLib* slib = gg->getSurfaceLib();
     993 
     994     // boundary identification via 4-uint 
     995     unsigned int boundary = blib->addBoundary(
     996                                                mt_p->getShortName(),
     997                                                osurf ? osurf->getShortName() : NULL ,
     998                                                isurf ? isurf->getShortName() : NULL ,
     999                                                mt->getShortName()
    1000                                              );
    1001 
    1002     solid->setBoundary(boundary);
    ....
    1019     if(m_volnames)
    1020     {
    1021         solid->setPVName(pv);
    1022         solid->setLVName(lv);
    1023     }
    ....
    1029     return solid ;
    1030 }

    ///
    ///       "boundary" int identifies unique combination of guint4 (imat,isur,osur,omat) indices 
    ///       and is assigned to the GSolid
    ///
    ///       how to reconstruct volume names for a surface post cache ?
    ///       
    ///       seems no way to know if skin or border ??
    ///       but are most interested in cathode SensorSurface 
    ///        ... which are logical skin surface (ie associated to only a few lv names
    ///       


Solids recursively collected into GGeo::

     802 void AssimpGGeo::convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent)
     803 {
     804     // recursive traversal of the AssimpNode tree
     805     // note that full tree is traversed even when a partial selection is applied 
     806 
     807 
     808     GSolid* solid = convertStructureVisit( gg, node, depth, parent);
     809 
     810     bool selected = m_selection && m_selection->contains(node) ;
     811 
     812     solid->setSelected(selected);
     813 
     814     gg->add(solid);
     815 
     816     if(parent) // GNode hookup
     817     {
     818         parent->addChild(solid);
     819         solid->setParent(parent);
     820     }
     821     else
     822     {
     823         assert(node->getIndex() == 0);   // only root node has no parent 
     824     }
     825 
     826     for(unsigned int i = 0; i < node->getNumChildren(); i++) convertStructure(gg, node->getChild(i), depth + 1, solid);
     827 }
     828 


::

     873 void GGeo::add(GSolid* solid)
     874 {
     875     m_solids.push_back(solid);
     876     unsigned int index = solid->getIndex(); // absolute node index, independent of the selection
     877     //printf("GGeo::add solid %u \n", index);
     878     m_solidmap[index] = solid ;
     879 
     880     if(m_volnames)
     881     {
     882         m_lvlist->add(solid->getLVName());
     883         m_pvlist->add(solid->getPVName());
     884     }
     885 
     886     GSolid* check = getSolid(index);
     887     assert(check == solid);
     888 }


GSolid(GNode) are persisted into GMergedMesh(GMesh)::

     596 void GMesh::allocate()
     597 {
     598 
     599     unsigned int numVertices = getNumVertices();
     600     unsigned int numFaces = getNumFaces();
     601     unsigned int numSolids = getNumSolids(); 
     ...
     ///
     ///  lots of solid level info in GMesh
     ///
     627     setCenterExtent(new gfloat4[numSolids]);
     628     setBBox(new gbbox[numSolids]);
     629     setMeshes(new unsigned int[numSolids]);
     630     setNodeInfo(new guint4[numSolids]);          //  nface/nvert/nodeIndex/parentIndex
     631     setIdentity(new guint4[numSolids]);          //  node/mesh/boundary/sensor 
     632     setTransforms(new float[numSolids*16]);
     633 
     634     LOG(info) << "GMesh::allocate DONE " ;
     635 }



GMergedMesh/0/identity.npy ana/mergedmesh.py::

    In [1]: mm
    Out[1]: 
               aiidentity : (1, 1, 4) 
              itransforms : (1, 4, 4) 

                     bbox : (12230, 6)    ## numSolids
            center_extent : (12230, 4) 
                   meshes : (12230, 1) 
                 nodeinfo : (12230, 4)    ## nface/nvert/nodeIndex/parentIndex
                 identity : (12230, 4)    ## nodeIndex/mesh/boundary/sensor 
                iidentity : (12230, 4) 
               transforms : (12230, 16) 

                    nodes : (434816, 1)    ## numFaces
               boundaries : (434816, 1) 
                  sensors : (434816, 1) 
                  indices : (1304448, 1)  ## 434816*3   "faces" 

                 vertices : (225200, 3)      ## numVertices
                  normals : (225200, 3) 
                   colors : (225200, 3) 



    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/identity.npy")

    In [3]: a.shape
    Out[3]: (12230, 4)

    In [4]: a
    Out[4]: 
    array([[    0,   248,     0,     0],
           [    1,   247,     1,     0],
           [    2,    21,     2,     0],
           ..., 
           [12227,   243,   122,     0],
           [12228,   244,   122,     0],
           [12229,   245,   122,     0]], dtype=uint32)

     


Ancient GDML Export has no surfaces OR optical props (is also has no material properties)
-------------------------------------------------------------------------------------------

::

    delta:DayaBay_VGDX_20140414-1300 blyth$ grep surface /tmp/g4_00.gdml 
    delta:DayaBay_VGDX_20140414-1300 blyth$ grep optical /tmp/g4_00.gdml 
    delta:DayaBay_VGDX_20140414-1300 blyth$ grep EFFICIENCY /tmp/g4_00.gdml 
    delta:DayaBay_VGDX_20140414-1300 blyth$ 


CGDMLDetector::addMPT
------------------------

Ancient GDML has materials, but they have no properties...  
Added them from the G4DAE/ggeo material library::


    097 void CGDMLDetector::addMPT()
     98 {
     99     // GDML exported by geant4 that comes with nuwa lack material properties 
    100     // so use the properties from the G4DAE export 
    101 
    ///
    122     unsigned int ng4mat = m_traverser->getNumMaterialsWithoutMPT() ;
    123     for(unsigned int i=0 ; i < ng4mat ; i++)
    124     {
    125         G4Material* g4mat = m_traverser->getMaterialWithoutMPT(i) ;
    126         const char* name = g4mat->GetName() ;
    127 
    128         std::vector<std::string> elem;
    129         boost::split(elem,name,boost::is_any_of("/"));
    130         assert(elem.size() == 4 && "expecting material names like /dd/Materials/GdDopedLS " );
    131         const char* shortname = elem[3].c_str();
    132 
    133         const GMaterial* ggmat = m_lib->getMaterial(shortname);
    134         assert(ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material") ;
    135 
    136         LOG(debug) << "CGDMLDetector::addMPT"
    137                   << " g4mat " << std::setw(45) << name
    138                   << " shortname " << std::setw(25) << shortname
    139                    ;
    140 
    141         G4MaterialPropertiesTable* mpt = m_lib->makeMaterialPropertiesTable(ggmat);
    ///
    ///              CPropLib::makeMaterialPropertiesTable  converts ggeo material into G4 MPT 
    ///
    142         g4mat->SetMaterialPropertiesTable(mpt);
    143         //m_lib->dumpMaterial(g4mat, "CGDMLDetector::addMPT");        
    144 
    145     }


CGDMLDetector::addSurfaces ?
-------------------------------

* looks like the ancient GDML geometry lacks surfaces entirely 

Questions:

* are the volume names including the pointers between G4DAE and GDML matching
  (they should be the GDML and G4DAE were exported from the same process)
  
  * they are for the cathodes 


* vague recall that CTestDetector used BorderSurface in order pin down the 
  photon direction to detect, for this need to have the pvnames from a tree traverse 
  (see GGeoTest for this) 


NEXT
------


Add methods like below to GGeo, like in GGeoTest::

    //
    //    private:
    //        void findSensorVolumePairs();
    //    public:
    //        unsigned getNumSensorVolumePairs();
    //        const std::pair<std::string, std::string>& getSensorVolumePair(unsigned p);    


Use the pairs in CGeometry to reconstruct G4LogicalBorderSurface for the cathodes
when using CGDMLDetector.

Avoid duplicated geometry loading in CProplib 





Code Trace photon SD flags
----------------------------


optixrap- where flags come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

oxrap/cu/genrate.cu::

    402 
    403         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    404         {
    405             command = propagate_at_surface(p, s, rng);
    406             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    407             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    408         }
    409         else
    410         {
    411             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    412             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    413             // tacit CONTINUE
    414         }


oxrap/cu/propagate.h::

    455 /*
    456 propagate_at_surface
    457 ======================
    458 
    459 Inputs:
    460 
    461 * s.surface.x detect
    462 * s.surface.y absorb              (1.f - reflectivity ) ?
    463 * s.surface.z reflect_specular
    464 * s.surface.w reflect_diffuse
    ...
    488 __device__ int
    489 propagate_at_surface(Photon &p, State &s, curandState &rng)
    490 {
    491 
    492     float u = curand_uniform(&rng);
    493 
    494     if( u < s.surface.y )   // absorb   
    495     {
    496         s.flag = SURFACE_ABSORB ;
    497         return BREAK ;
    498     }
    499     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    500     {
    501         s.flag = SURFACE_DETECT ;
    502         return BREAK ;
    503     }
    504     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    505     {
    506         s.flag = SURFACE_DREFLECT ;
    507         propagate_at_diffuse_reflector(p, s, rng);
    508         return CONTINUE;
    509     }
    510     else
    511     {
    512         s.flag = SURFACE_SREFLECT ;
    513         propagate_at_specular_reflector(p, s, rng );
    514         return CONTINUE;
    515     }
    516 }





* surface handling requires > 0 surface index


optixrap where properties come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    240 GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>* src)
    241 {
    ...
    258         GOpticalSurface* os = src->getOpticalSurface() ;  // GSkinSurface and GBorderSurface ctor plant the OpticalSurface into the PropertyMap
    259 
    260         if(src->isSensor())
    261         {
    262             GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY);
    263             assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );
    264 
    265             if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
    266             {
    267                 _detect           = makeConstantProperty(m_fake_efficiency) ;
    268                 _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
    269                 _reflect_specular = makeConstantProperty(0.0);
    270                 _reflect_diffuse  = makeConstantProperty(0.0);
    271             }
    272             else
    273             {
    274                 _detect = _EFFICIENCY ;
    275                 _absorb = GProperty<float>::make_one_minus( _detect );
    276                 _reflect_specular = makeConstantProperty(0.0);
    277                 _reflect_diffuse  = makeConstantProperty(0.0);
    278             }
    279         }
    280         else
    281         {
    282             GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY);
    283             assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
    284 
    285             if(os->isSpecular())
    286             {
    287                 _detect  = makeConstantProperty(0.0) ;
    288                 _reflect_specular = _REFLECTIVITY ;
    289                 _reflect_diffuse  = makeConstantProperty(0.0) ;
    290                 _absorb  = GProperty<float>::make_one_minus(_reflect_specular);
    291             }
    292             else
    293             {
    294                 _detect  = makeConstantProperty(0.0) ;
    295                 _reflect_specular = makeConstantProperty(0.0) ;
    296                 _reflect_diffuse  = _REFLECTIVITY ;
    297                 _absorb  = GProperty<float>::make_one_minus(_reflect_diffuse);
    298             }
    299         }
    300     }





CFG4 Where the flags come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cfg4/OpStatus.hh::

    020 CFG4_API unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst);

cfg4/OpStatus.cc::

    207 unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst)
    208 {
    209     G4StepStatus status = point->GetStepStatus()  ;
    210     // TODO: cache the relevant process objects, so can just compare pointers ?
    211     const G4VProcess* process = point->GetProcessDefinedStep() ;
    212     const G4String& processName = process ? process->GetProcessName() : "NoProc" ;
    213 
    214     bool transportation = strcmp(processName,"Transportation") == 0 ;
    215     bool scatter = strcmp(processName, "OpRayleigh") == 0 ;
    216     bool absorption = strcmp(processName, "OpAbsorption") == 0 ;
    217 
    218     unsigned int flag(0);
    219     if(absorption && status == fPostStepDoItProc )
    220     {
    221         flag = BULK_ABSORB ;
    222     }
    223     else if(scatter && status == fPostStepDoItProc )
    224     {
    225         flag = BULK_SCATTER ;
    226     }
    227     else if(transportation && status == fWorldBoundary )
    228     {
    229         flag = SURFACE_ABSORB ;   // kludge for fWorldBoundary - no surface handling yet 
    230     }
    231     else if(transportation && status == fGeomBoundary )
    232     {
    233         flag = OpBoundaryFlag(bst) ; // BOUNDARY_TRANSMIT/BOUNDARY_REFLECT/NAN_ABORT/SURFACE_ABSORB/SURFACE_DETECT
    234     }
    235     return flag ;
    236 }


    144 unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status)  ///   non-API private function
    145 {
    146     unsigned int flag = 0 ;
    147     switch(status)
    148     {
    149         case FresnelRefraction:
    150                                flag=BOUNDARY_TRANSMIT;
    151                                break;
    152         case TotalInternalReflection:
    153         case       FresnelReflection:
    154                                flag=BOUNDARY_REFLECT;
    155                                break;
    156         case StepTooSmall:
    157                                flag=NAN_ABORT;
    158                                break;
    159         case Absorption:
    160                                flag=SURFACE_ABSORB ;
    161                                break;
    162         case Detection:
    163                                flag=SURFACE_DETECT ;
    164                                break;
    165         case Undefined:
    166         case Transmission:
    167         case BackScattering:


G4 Where Detection flags come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


g4-;g4-cls G4OpBoundaryProcess::

    306 inline
    307 void G4OpBoundaryProcess::DoAbsorption()
    308 {
    309               theStatus = Absorption;
    310 
    311               if ( G4BooleanRand(theEfficiency) ) {
    312 
    313                  // EnergyDeposited =/= 0 means: photon has been detected
    314                  theStatus = Detection;
    315                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    316               }
    317               else {
    318                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    319               }
    320 
    321               NewMomentum = OldMomentum;
    322               NewPolarization = OldPolarization;
    323 
    324 //              aParticleChange.ProposeEnergy(0.0);
    325               aParticleChange.ProposeTrackStatus(fStopAndKill);
    326 }


    165 G4VParticleChange*
    166 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    167 {
    168         theStatus = Undefined;     
    ///
    ///    DoAbsorption gets called for each of the boundary types...
    ///    coming up with "Detection" requires luck and a suitable theEfficiency value
    ///
    ///
    483         else if (type == dielectric_dielectric) {
    484 
    485           if ( theFinish == polishedbackpainted ||
    486                theFinish == groundbackpainted ) {
    487              DielectricDielectric();
    488           }
    489           else {
    490              G4double rand = G4UniformRand();
    491              if ( rand > theReflectivity ) {
    492                 if (rand > theReflectivity + theTransmittance) {
    493                    DoAbsorption();
    494                 } else {
    495                    theStatus = Transmission;
    496                    NewMomentum = OldMomentum;
    497                    NewPolarization = OldPolarization;
    498                 }
    499              }


Volume boundary needs G4LogicalBorderSurface or G4LogicalSkinSurface with MPT to provide non-zero EFFICIENCY::

     337     if (Surface) OpticalSurface =
     338            dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
     339 
     340     if (OpticalSurface) {
     341 
     342            type      = OpticalSurface->GetType();
     343        theModel  = OpticalSurface->GetModel();
     344        theFinish = OpticalSurface->GetFinish();
     345 
     346        aMaterialPropertiesTable = OpticalSurface->
     347                     GetMaterialPropertiesTable();
     348 
     349            if (aMaterialPropertiesTable) {
     ...
     ... 
     387               PropertyPointer =
     388               aMaterialPropertiesTable->GetProperty("EFFICIENCY");
     389               if (PropertyPointer) {
     390                       theEfficiency =
     391                       PropertyPointer->Value(thePhotonMomentum);



So it looks like the CGDMLDetector is missing Optical Surfaces whereas the CTestDetector has them ?




