surface_review
================

Surface Overhaul Approach
---------------------------

* Review the various surface info flows

* find sticking-points/workarounds/kludges

Some things were done in an expedient manner 
in order to focus on part of the chain. As having 
full chain in flux is too much to handle.

* **NOW** IS THE TIME TO FIX THESE THINGS.

* eg the border/skin info is available pre-cache, so kludging 
  reconstruction of that post-cache points to infoloss
  in persisting and the GSurLib workaround that got out of control 



Hmm GDML reconstruction does nothing special : BUT there are ptrs on names
----------------------------------------------------------------------------

* TODO: confirm assumption that ptr in pv names mean they are all unique ? 
  (ie different name for each traversal index)

  * for lv : i guess no, lv are recycled greatly 


* :doc:`surface_review_gdml`

::

    simon:ggeo blyth$ opticks-find setPVName
    ./assimprap/AssimpGGeo.cc:        solid->setPVName(pv);
    ./ggeo/GMaker.cc:    solid->setPVName( strdup(pvn.c_str()) );
    ./ggeo/GScene.cc:    node->setPVName( pvname.c_str() );
    ./ggeo/GSolid.cc:void GSolid::setPVName(const char* pvname)
    ./ggeo/GSolid.hh:      void setPVName(const char* pvname);
    simon:opticks blyth$ 




Improved PropLib Persisting with JSON metadata
----------------------------------------------------

Added NMeta json infrastructure to enable full fidelity 
metadata to be stored with persisted PropLib. 

Where to tack the metadata ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding surfaces/materials is not a common thing to do, so:
 
* global metadata for the entire PropLibs, dict-of-dict style 
  top level keys being the material/surface names 

* current NParameters uses BList string,string persisting 
  which restricts it to a single level

* developed NMeta using nlohmann::json for more flexible metadata
 
* placed m_meta into GPropertyMap/GPropertyLib, the maps correpond to 
  individual surf/mat etc.. and the libs to collections of those :
  NMeta is composable allowing the lib to amalgamate all meta data 
  prior to save and then distribute it on load  


New metadata infrastructure operational via geocache::

    simon:ggeo blyth$ op --surf 6
    ...
    2017-11-07 21:12:30.368 INFO  [3558034] [GPropertyMap<float>::dumpMeta@146] GSurfaceLib::dump.index
    2017-11-07 21:12:30.368 INFO  [3558034] [NMeta::dump@74] {
        "index": 6,
        "name": "lvPmtHemiCathodeSensorSurface",
        "shortname": "lvPmtHemiCathodeSensorSurface",
        "sslv": "__dd__Geometry__PMT__lvPmtHemiCathode0xc2cdca0",
        "type": "surface"
    }
    /Users/blyth/opticks/bin/op.sh RC 0
    simon:ggeo blyth$ 
    simon:ggeo blyth$ 



Vague Recollection of the history of this..
---------------------------------------------

Originally (whilst focus was entirely on OptiX geometry) 
materials and surfaces were not persisted to geocache, 
instead the boundary lib comprising all the interleaved props was persisted alone.

Subsequently the need for dynamic boundaries for testing meant that moved to 
the boundary buffer tex being dynamically derived from integers representing 
materials and surfaces, and added PropLib persisting then.

The thing is that OptiX does not need the border/skin surface volume names
because the info is already present in the form of the boundary indices that
are affixed to every piece of geometry. These boundary spec being formed pre-cache
whilst the info is available.

Subsequently cfg4 means need to reconstitute the G4 border/skin objects. Although 
it is in principal possible to disentangle these from the boundaries, 
it aint at all simple : resulting in complex workarounds in GSurLib/CSurLib/GSur/...

Solution: improve the GPropLib persisting with some NMeta metadata 
so that the G4 geometry can be reconstructed without jumping thru hoops. 
Going back a few steps avoids the complexity of operating at just the last step.



Potential Missing Surfaces ?
--------------------------------------

Do the original G4 border/skin surfaces survive the journey ? 

* TODO: a full geometry workflow test, but with a simple enough geometry to illustrate the issues 

* assumption of one location for a named PV pair is incorrect ?
  (because node graph, not tree)

* using traversal indices may be a way to flatten the graph into a tree
  and avoid the issue

Perhaps : 

* get all the traversal indices of each PV name 
  and then do parent/child relation checks to reconstruct 
  valid border surface pairs ?

* for skin surfaces using logical name lookup should be ok

PV Addressing
---------------

* CTraverser.m_pvnames are without the ptr

::


    (lldb) p m_pvnames
    (std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > > >) $3 = size=12230 {
      [0] = "World_PV"
      [1] = "/dd/Structure/Sites/db-rock"
      [2] = "/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop"
      [3] = "/dd/Geometry/Sites/lvNearHallTop#pvNearTopCover"
      [4] = "/dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1"
      [5] = "/dd/Geometry/RPC/lvRPCMod#pvRPCFoam"


    (lldb) p name
    (const char *) $4 = 0x000000010b2d7f60 "__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018"
    (lldb) p BStr::DAEIdToG4(name)
    (char *) $5 = 0x000000010b2d8ff0 "/dd/Geometry/Sites/lvNearHallBot--pvNearPoolDead"



GSurfaceLib model
------------------

When a set of surface props are attached at multiple locations 
(bpv1/bpv2 pairs or sslv) then the surface must be repeated.

* ie surface identity incorporates location 



CSurLib instanciated by CDetector::attachSurfaces from CGeometry::init
-------------------------------------------------------------------------

::

    267 void CDetector::attachSurfaces()
    268 {
    269     // invoked from CGeometry::init immediately after CTestDetector or GDMLDetector instanciation
    270 
    271     if(m_dbgsurf)
    272         LOG(info) << "[--dbgsurf] CDetector::attachSurfaces START closing gsurlib, creating csurlib  " ;
    273 
    274     m_gsurlib->close();
    275 
    276     m_csurlib = new CSurLib(m_gsurlib);
    277 
    278     m_csurlib->convert(this);
    279 
    280     if(m_dbgsurf)
    281         LOG(info) << "[--dbgsurf] CDetector::attachSurfaces DONE " ;
    282 
    283 }
    284 


     56 void CGeometry::init()
     57 {
     58     CDetector* detector = NULL ; 
     59     if(m_ok->hasOpt("test"))
     60     {
     61         LOG(fatal) << "CGeometry::init G4 simple test geometry " ; 
     62         OpticksQuery* query = NULL ;  // normally no OPTICKS_QUERY geometry subselection with test geometries
     63         detector  = static_cast<CDetector*>(new CTestDetector(m_hub, query)) ;
     64     }
     65     else
     66     {
     67         // no options here: will load the .gdml sidecar of the geocache .dae 
     68         LOG(fatal) << "CGeometry::init G4 GDML geometry " ;
     69         OpticksQuery* query = m_ok->getQuery();
     70         detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query)) ;
     71     }
     72 
     73     detector->attachSurfaces();
     74 
     75     m_detector = detector ;
     76     m_lib = detector->getPropLib();
     77 }




Surface Info Flows
-----------------------

GGeo : Full Triangulated Geometry Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* export of G4 border/skin surfaces into COLLADA G4DAE

* AssimpGGeo parsing of G4DAE into GGeo/GSurfaceLib 

* persisting GGeo/GSurfaceLib to geocache

* loading GGeo/GSurfaceLib from geocache

* translation of loaded GGeo/GSurfaceLib into OptiX geometry 

* translation of loaded GGeo/GSurfaceLib into Geant4 geometry 


GScene : Full Analytic Geometry Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* export G4 border/skin into GDML together with everything else

  * NB GDML looses some material/surf info, so the GDML flow is
    not standalone (even in current GDML, let alone some old GDML exports 
    that are still supporting)... So it needs to be used together with G4DAE

* python GDML parsing into GLTF json 

* NGLTF/NScene/GScene parsing of GLTF, yielding GScene/GSurfaceLib

* FROM GSurfaceLib the story is the same as above


GGeoTest : Test Geometry Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* parse NCSG python buffers into NCSGList of trees, including txt
  files with boundary specification for each solid

* construction of GGeoTest geometry from NCSG, the surfaces 
  referred to by name within the boundary specification


Fundamental surface difference between full/test geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice the fundamental difference wrt surfaces between full and test geometries, 

* full geometries have original "truth" sslv,bspv1,bspv2 names
  locating the surfaces which are NOW passed forward from the G4 geometry 
  into GGeo/GSurfaceLib(GPropLib) using NMeta/json to survive the geocache 
  this should allow simple reconstruction of a G4 geometry from the GGeo one  

* hmm : what about surface identity, presumably this means there is duplication
  of same surface properties into different locations ?

* test geometries must create "truth" regarding surface locations as they go along
   
  * base geometry surfaces are referenced for their properties, NOT LOCATIONS 

  * locations specified by base geometry sslv/bspv1/bspv2 names are 
    not applicable to test geometries which have entirely different names for the volumes


GSurfaceLib -> CSurfaceLib translation of both full and test geometries ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suspect easiest to make test geometry to look just like full geometry
as soon as possible by giving them the requisite metadata names.

Perhaps:

* dynamically apply modifications to base surface locations 
* when are the names coming from (GMaker ?) 


GSurfaceLib::save
--------------------

::


    051 void GSurfaceLib::save()
     52 {
     53     saveToCache();
    ///  from GPropertyLib::saveToCache
     54     saveOpticalBuffer();
     55 }
                  
     73 void GSurfaceLib::saveOpticalBuffer()
     74 {   
     75     NPY<unsigned int>* ibuf = createOpticalBuffer();
     76     saveToCache(ibuf, "Optical") ;
     77     setOpticalBuffer(ibuf);
     78 }


    418 void GPropertyLib::saveToCache()
    419 {
    420 
    421     LOG(trace) << "GPropertyLib::saveToCache" ;
    422 
    423 
    424     if(!isClosed()) close();
    425 
    426     if(m_buffer)
    427     {
    428         std::string dir = getCacheDir();
    429         std::string name = getBufferName();
    430         m_buffer->save(dir.c_str(), name.c_str());
    431     }
    432 
    433     if(m_names)
    434     {
    435         m_names->save(m_resource->getIdPath());
    436     }
    437 
    438     LOG(trace) << "GPropertyLib::saveToCache DONE" ;
    439 
    440 }


GSurLib formerly of GGeo, now moved to OpticksHub
------------------------------------------------------

Aiming to eliminate GSurLib, as: 

* overcomplicated 

* only used by CSurLib

* the original purpose of distinguishing skin from border surfaces
  from their pattern of use : turned out not to be possible


CDetector
------------

::

    036 CDetector::CDetector(OpticksHub* hub, OpticksQuery* query)
     37   :
     38   m_hub(hub),
     39   m_ok(m_hub->getOpticks()),
     40   m_ggeo(m_hub->getGGeo()),
     41   m_blib(new CBndLib(m_hub)),
     42   m_gsurlib(m_hub->getSurLib()),   // invokes the deferred GGeo::createSurLib  
     43   m_csurlib(NULL),

    621 GSurLib* OpticksHub::getSurLib()
    622 {
    623     return m_ggeo ? m_ggeo->getSurLib() : NULL ;
    624 }



GSurLib
-----------

::

    GSurLib* OpticksHub::createSurLib(GGeoBase* ggb)
    {
        GSurLib* gsl = new GSurLib(m_ok, ggb );  
        return gsl ; 
    }

    GSurLib* OpticksHub::getSurLib()
    {
        if( m_gsurlib == NULL )
        {   
            // this method motivating making GGeoTest into a GGeoBase : ie standard geo provider
            GGeoBase* ggb = getGGeoBase();    // three-way choice 
            m_gsurlib = createSurLib(ggb) ;
        }   
        return m_gsurlib ; 
    }




AssimpGGeo::convertMaterials adding to GGeo/GSurfaceLib
-----------------------------------------------------------

Assimp has no "surface" so aiMaterials are used to hold both surface and material 
info with g4dae extra properties to distinguish

::

     392         LOG(debug) << "AssimpGGeo::convertMaterials " << i << " " << name ;
     393 
     394         const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
     395         const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );
     396 
     397         const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );
     398 
     399         const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
     400         const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
     401         const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
     402         const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
     403         const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );
     404 
     405 
     406         GOpticalSurface* os = osnam && ostyp && osmod && osfin && osval ? new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) : NULL ;
     407 
     408 
     409         // assimp "materials" are used to hold skinsurface and bordersurface properties, 
     410         // as well as material properties
     411         // which is which is determined by the properties present 
     412 
     413         if(os)
     414         {
     415             LOG(debug) << "AssimpGGeo::convertMaterials os " << i << " " << os->description();
     416 
     417             // assert(strcmp(osnam, name) == 0); 
     418             //      formerly enforced same-name convention between OpticalSurface 
     419             //      and the skin or border surface that references it, but JUNO doesnt follow that  
     420         }
     421 
     422         if( sslv )
     423         {
     424             assert(os && "all ss must have associated os");
     425 
     426             GSkinSurface* gss = new GSkinSurface(name, index, os);
     427 
     428 
     429             LOG(debug) << "AssimpGGeo::convertMaterials GSkinSurface "
     430                       << " name " << name
     431                       << " sslv " << sslv
     432                       ;
     433 
     434             gss->setStandardDomain(standard_domain);
     435             gss->setSkinSurface(sslv);
     436             addProperties(gss, mat );
     437 
     438             LOG(debug) << gss->description();
     439             gg->add(gss);
     440 
     441             {
     442                 // without standard domain applied
     443                 GSkinSurface*  gss_raw = new GSkinSurface(name, index, os);
     444                 gss_raw->setSkinSurface(sslv);
     445                 addProperties(gss_raw, mat );
     446                 gg->addRaw(gss_raw);  // this was erroreously gss for a long time
     447             }
     448 
     449         }
     450         else if (bspv1 && bspv2 )
     451         {
     452             assert(os && "all bs must have associated os");
     453             GBorderSurface* gbs = new GBorderSurface(name, index, os);
     454 
     455             gbs->setStandardDomain(standard_domain);
     456             gbs->setBorderSurface(bspv1, bspv2);
     457             addProperties(gbs, mat );
     458 
     459             LOG(debug) << gbs->description();
     460 
     461             gg->add(gbs);
     462 
     463             {
     464                 // without standard domain applied
     465                 GBorderSurface* gbs_raw = new GBorderSurface(name, index, os);
     466                 gbs_raw->setBorderSurface(bspv1, bspv2);
     467                 addProperties(gbs_raw, mat );
     468                 gg->addRaw(gbs_raw);
     469             }
     470         }
     471         else
     472         {
     473             assert(os==NULL);
     474 
     475 
     476             //printf("AssimpGGeo::convertMaterials aiScene materialIndex %u (GMaterial) name %s \n", i, name);
     477             GMaterial* gmat = new GMaterial(name, index);
     478             gmat->setStandardDomain(standard_domain);
     479             addProperties(gmat, mat );
     480             gg->add(gmat);
     481 
     482             {
     483                 // without standard domain applied
     484                 GMaterial* gmat_raw = new GMaterial(name, index);
     485                 addProperties(gmat_raw, mat );
     486                 gg->addRaw(gmat_raw);
     487             }
     488 
     489             if(hasVectorProperty(mat, EFFICIENCY ))
     490             {
     491                 assert(gg->getCathode() == NULL && "only expecting one material with an EFFICIENCY property" );
     492                 gg->setCathode(gmat) ;
     493                 m_cathode = mat ;
     494             }




GSurfaceLib::add
-------------------

::

    202 void GSurfaceLib::add(GBorderSurface* raw)
    203 {
    204     GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
    205     add(surf);
    206 }
    207 void GSurfaceLib::add(GSkinSurface* raw)
    208 {
    209     LOG(trace) << "GSurfaceLib::add(GSkinSurface*) " << ( raw ? raw->getName() : "NULL" ) ;
    210     GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
    211     add(surf);
    212 }
    213 
    214 void GSurfaceLib::add(GPropertyMap<float>* surf)
    215 {
    216     assert(!isClosed());
    217 
    218     GPropertyMap<float>* ssurf = createStandardSurface(surf) ;
    219 
    220     addDirect(ssurf);
    221 }
    222 
    223 
    224 void GSurfaceLib::addDirect(GPropertyMap<float>* surf)
    225 {
    226     assert(!isClosed());
    227     m_surfaces.push_back(surf);
    228 }




GSurfaceLib in geocache
--------------------------

No json or txt with the surfacelib::

    simon:GSurfaceLib blyth$ ll
    total 128
    drwxr-xr-x   4 blyth  staff    136 Jul  3 15:04 .
    drwxr-xr-x  19 blyth  staff    646 Aug 29 10:46 ..
    -rw-r--r--   1 blyth  staff    848 Aug 30 13:35 GSurfaceLibOptical.npy
    -rw-r--r--   1 blyth  staff  59984 Aug 30 13:35 GSurfaceLib.npy
    simon:GSurfaceLib blyth$ 




