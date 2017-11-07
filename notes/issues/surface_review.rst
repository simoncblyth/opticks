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


Improved PropLib Persisting with JSON metadata
----------------------------------------------------

* adding a GItemList txt list string to every surface carying 
  some metadata OR even adopting some standard naming  

Added NJS json infrastructure to enable full fidelity 
metadata to be stored with persisted PropLib. 

Where to tack the metadata ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding surfaces/materials is not a common thing to do, so:
 
* global metadata for the entire PropLibs, dict-of-dict style 
  top level keys being the material/surface names 


* uses NParameters ? That already has BList string,string persisting 




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

Solution: improve the PropLib persisting with some metadata so that the G4 geometry 
can be reconstructed without jumping thru hoops. Going back a few steps avoids the 
complexity of operating at just the last step.



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

* python GDML parsing into GLTF json

* NGLTF/NScene/GScene parsing of GLTF, yielding GScene/GSurfaceLib

* FROM GSurfaceLib the story is the same as above


GGeoTest : Test Geometry Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* parse NCSG python buffers into NCSGList of trees, including txt
  files with boundary specification for each solid

* construction of GGeoTest geometry from NCSG, the surfaces 
  referred to by name within the boundary specification



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




