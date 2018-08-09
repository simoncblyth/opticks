ab-surf1
===========

FIXED
------

::

    epsilon:issues blyth$ o
    M assimprap/AssimpGGeo.cc
    M bin/ab.bash
    M cfg4/CSurfaceLib.cc
    M extg4/X4OpticalSurface.cc
    M extg4/X4PhysicalVolume.cc
    M ggeo/GOpticalSurface.cc
    M ggeo/GOpticalSurface.hh
    M ggeo/GPropertyMap.cc
    M sysrap/SDigest.cc
    M sysrap/SDigest.hh
    A notes/issues/ab-surf1.rst
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ hg commit -m "fix ab-surf1 issue with optical surface values, was mixup regarding fractional or percent storage "
    epsilon:opticks blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/opticks
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 1 changesets with 11 changes to 11 files
    epsilon:opticks blyth$ 


Probable cause
---------------

Recall that B boots from GDML, with fixups for missing surfaces...
coming from G4DAE : looks like the fixup of Optical surfaces 
not working.


Issue : in B most surfaces have 4th column value of 0 
--------------------------------------------------------

::

    epsilon:issues blyth$ ab-surf1
    ...
    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-surf1.py
    a.shape : (48, 2, 39, 4) 
    b.shape : (48, 2, 39, 4) 
    ab.max() 5.9604645e-08
     abmx*1000000000.0 : max absolute difference of property values of each surface 
          0.000  : ESRAirSurfaceTop 
          0.000  : ESRAirSurfaceBot 
          0.000  : SSTOilSurface 
          0.000  : SSTWaterSurfaceNear1 
          0.000  : SSTWaterSurfaceNear2 
    ...
          0.000  : NearOutInPiperSurface 
          0.000  : NearOutOutPiperSurface 
          0.000  : LegInDeadTubSurface 
          0.000  : perfectDetectSurface 
          0.000  : perfectAbsorbSurface 
          0.000  : perfectSpecularSurface 
          0.000  : perfectDiffuseSurface 
         59.605  : lvHeadonPmtCathodeSensorSurface 
         59.605  : lvPmtHemiCathodeSensorSurface 
    np.hstack( [oa, ob] )
    [[  0   0   0   0   0   0   0   0]
     [  1   0   0   0   1   0   0   0]
     [  2   0   3 100   2   0   3   0]
     [  3   0   3 100   3   0   3   0]
     [  4   0   3 100   4   0   3   0]
     [  5   0   3  20   5   0   3   0]
     [  6   0   3  20   6   0   3   0]
     [  7   0   3  20   7   0   3   0]
     [  8   0   3 100   8   0   3   0]
     [  9   0   3 100   9   0   3   0]
     [ 10   0   3 100  10   0   3   0]
     [ 11   0   3 100  11   0   3   0]
    ...
     [ 40   0   3 100  40   0   3   0]
     [ 41   0   3 100  41   0   3   0]
     [ 42   1   1 100  42   1   1 100]      ## perfect 
     [ 43   1   1 100  43   1   1 100]
     [ 44   1   1 100  44   1   1 100]
     [ 45   1   1 100  45   1   1 100]
     [ 46   0   3 100  46   0   3 100]      ## sensor
     [ 47   0   3 100  47   0   3 100]]


See ab-surf1 for the script::

    1047 oa = a_npy("GSurfaceLib/GSurfaceLibOptical.npy")
    1048 ob = b_npy("GSurfaceLib/GSurfaceLibOptical.npy")
    1049 assert oa.shape == ob.shape
    1050 
    1051 s = "np.hstack( [oa, ob] )"
    1052 print s
    1053 print eval(s)



GSurfaceLib
--------------

Difference in values in GSurfaceLibOptical::

    0105 void GSurfaceLib::saveOpticalBuffer()
     106 {
     107     NPY<unsigned int>* ibuf = createOpticalBuffer();
     108     saveToCache(ibuf, "Optical") ;
     109     setOpticalBuffer(ibuf);
     110 }

    1021 NPY<unsigned int>* GSurfaceLib::createOpticalBuffer()
    1022 {
    1023     std::vector<guint4> optical ;
    1024     unsigned int ni = getNumSurfaces();
    1025     for(unsigned int i=0 ; i < ni ; i++) optical.push_back(getOpticalSurface(i));
    1026     return createUint4Buffer(optical);
    1027 }

    0488 guint4 GSurfaceLib::createOpticalSurface(GPropertyMap<float>* src)
     489 {
     490    assert(src->isSkinSurface() || src->isBorderSurface() || src->isTestSurface());
     491    GOpticalSurface* os = src->getOpticalSurface();
     492    assert(os && "all skin/boundary surface expected to have associated OpticalSurface");
     493    guint4 optical = os->getOptical();
     494    return optical ;
     495 }
     496 
     497 guint4 GSurfaceLib::getOpticalSurface(unsigned int i)
     498 {
     499     GPropertyMap<float>* surf = getSurface(i);
     500     guint4 os = createOpticalSurface(surf);
     501     os.x = i ;
     502     return os ;
     503 }


GOpticalSurface
-----------------

::

    156 guint4 GOpticalSurface::getOptical()
    157 {
    158    guint4 optical ;
    159    optical.x = UINT_MAX ; //  place holder
    160    optical.y = boost::lexical_cast<unsigned int>(getType());
    161    optical.z = boost::lexical_cast<unsigned int>(getFinish());
    162 
    163    char* value = getValue();
    164    float percent = boost::lexical_cast<float>(value)*100.f ;   // express as integer percentage 
    165 
    166    unsigned upercent = unsigned(percent) ;   // rounds down 
    167   // unsigned upercent = boost::lexical_cast<unsigned int>(percent) ;
    168 
    169    optical.w = upercent ;
    170 
    171    return optical ;
    172 }



A : G4DAE route  
---------------------------------

G4DAEWriteStructure::OpticalSurfaceWrite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    242 /*
    243  * Create opticalsurface element with attributes from G4OpticalSurface*
    244  * append to first argument element
    245  * 
    246  * from G4GDMLWriteSolids::OpticalSurfaceWrite
    247  */
    248 void G4DAEWriteStructure::
    249 OpticalSurfaceWrite(xercesc::DOMElement* targetElement,
    250                     const G4OpticalSurface* const surf)
    251 {
    252    xercesc::DOMElement* optElement = NewElement("opticalsurface");
    253    G4OpticalSurfaceModel smodel = surf->GetModel();
    254    G4double sval = (smodel==glisur) ? surf->GetPolish() : surf->GetSigmaAlpha();
    255 
    256    optElement->setAttributeNode(NewNCNameAttribute("name", surf->GetName()));
    257    optElement->setAttributeNode(NewAttribute("model", smodel));
    258    optElement->setAttributeNode(NewAttribute("finish", surf->GetFinish()));
    259    optElement->setAttributeNode(NewAttribute("type", surf->GetType()));
    260    optElement->setAttributeNode(NewAttribute("value", sval));
    261 
    262    G4MaterialPropertiesTable* ptable = surf->GetMaterialPropertiesTable();
    263    PropertyWrite( optElement, ptable );
    264 
    265    targetElement->appendChild(optElement);
    266 }

AssimpGGeo::convertMaterials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strings from G4DAE are held in GOpticalSurfaces within GSkinSurface/GBorderSurface and collected 
in GSurfaceLib::

     430         const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
     431         const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
     432         const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
     433         const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
     434         const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );
     435 
     437         GOpticalSurface* os = osnam && ostyp && osmod && osfin && osval ? new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) : NULL ;
     ...
     453         if( sslv )
     454         {
     455             assert(os && "all ss must have associated os");
     456 
     457             GSkinSurface* gss = new GSkinSurface(name, index, os);
     458 


B : CGDMLDetector GDML parse and fixup followed by "direct" X4 conversion
-----------------------------------------------------------------------------

::

    298 void CDetector::attachSurfaces()
    299 {
    300     LOG(m_level) << "." ;
    301 
    302     int num_bs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
    303     int num_sk = G4LogicalSkinSurface::GetNumberOfSkinSurfaces();
    ...
    319     bool exclude_sensors = true ;
    320     m_slib->convert(this, exclude_sensors );


::

    079 void CSurfaceLib::convert(CDetector* detector, bool exclude_sensors)
     80 {
     ..
     90     setDetector(detector);
     91 
     92     unsigned num_surf = m_surfacelib->getNumSurfaces() ;
     ..
     99 
    100     for(unsigned i=0 ; i < num_surf ; i++)
    101     {  
    102         GPropertyMap<float>* surf = m_surfacelib->getSurface(i);
    103         const char* name = surf->getName();
    104         bool is_sensor_surface = GSurfaceLib::NameEndsWithSensorSurface( name ) ;
    105 
    106         if( is_sensor_surface && exclude_sensors )
    107         {
    108             LOG(error) << " skip sensor surf : "
    109                        << " name " << name
    110                        << " keys " << surf->getKeysString()
    111                        ;
    112             continue ;
    113         }
    114 
    115         if(surf->isBorderSurface())
    116         {
    117              G4OpticalSurface* os = makeOpticalSurface(surf);
    118              G4LogicalBorderSurface* lbs = makeBorderSurface(surf, os);
    119              m_border.push_back(lbs);
    120         }
    121         else if(surf->isSkinSurface())
    122         {
    123              G4OpticalSurface* os = makeOpticalSurface(surf);
    124              G4LogicalSkinSurface* lss = makeSkinSurface(surf, os);
    125              m_skin.push_back(lss);
    126         }
    127         else
    128         {



X4OpticalSurface::Convert
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     28 GOpticalSurface* X4OpticalSurface::Convert( const G4OpticalSurface* const surf )
     29 {
     ...
     79     const char* osnam = name ;
     80     const char* ostyp = BStr::itoa(type);
     81     const char* osmod = BStr::itoa(model);
     82     const char* osfin = BStr::itoa(finish);
     83     int percent = int(value*100.0) ;
     84     const char* osval = BStr::itoa(percent);
     85 
     86     GOpticalSurface* os = osnam && ostyp && osmod && osfin && osval ? new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) : NULL ;
     87     assert( os );
     88     return os ;
     89 }



A B dumps
-------------


::

    2018-08-09 18:24:20.373 ERROR [11494556] [X4LogicalBorderSurfaceTable::init@32]  NumberOfBorderSurfaces 8
    2018-08-09 18:24:20.373 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name               ESRAirSurfaceTop type 0 model 1 finish 0 value 0
    2018-08-09 18:24:20.373 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name               ESRAirSurfaceBot type 0 model 1 finish 0 value 0
    2018-08-09 18:24:20.373 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name                  SSTOilSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.373 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name           SSTWaterSurfaceNear1 type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.373 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name           SSTWaterSurfaceNear2 type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name          NearIWSCurtainSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name            NearOWSLinerSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name           NearDeadLinerSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [X4LogicalSkinSurfaceTable::init@32]  NumberOfSkinSurfaces num_src 34
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name           NearPoolCoverSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name                   RSOilSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name             AdCableTraySurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name            PmtMtTopRingSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name           PmtMtBaseRingSurface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.374 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name               PmtMtRib1Surface type 0 model 1 finish 3 value 0
    2018-08-09 18:24:20.375 ERROR [11494556] [*X4OpticalSurface::Convert@78]  name               PmtMtRib2Surface type 0 model 1 finish 3 value 0


After fix omission in CSurfaceLib::makeOpticalSurface::

    2018-08-09 19:04:05.307 ERROR [11521195] [X4LogicalBorderSurfaceTable::init@32]  NumberOfBorderSurfaces 8
    2018-08-09 19:04:05.307 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name               ESRAirSurfaceTop type 0 model 1 finish 0 value 0
    2018-08-09 19:04:05.307 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name               ESRAirSurfaceBot type 0 model 1 finish 0 value 0
    2018-08-09 19:04:05.307 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name                  SSTOilSurface type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.307 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name           SSTWaterSurfaceNear1 type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.307 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name           SSTWaterSurfaceNear2 type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.307 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name          NearIWSCurtainSurface type 0 model 1 finish 3 value 0.2
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name            NearOWSLinerSurface type 0 model 1 finish 3 value 0.2
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name           NearDeadLinerSurface type 0 model 1 finish 3 value 0.2
    2018-08-09 19:04:05.308 ERROR [11521195] [X4LogicalSkinSurfaceTable::init@32]  NumberOfSkinSurfaces num_src 34
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name           NearPoolCoverSurface type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name                   RSOilSurface type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name             AdCableTraySurface type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name            PmtMtTopRingSurface type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name           PmtMtBaseRingSurface type 0 model 1 finish 3 value 1
    2018-08-09 19:04:05.308 ERROR [11521195] [*X4OpticalSurface::Convert@78]  name               PmtMtRib1Surface type 0 model 1 finish 3 value 1
    ...


    2018-08-09 18:32:14.074 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                       __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop ostyp 0 osmod 1 osfin 0 osval 0
    2018-08-09 18:32:14.074 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                       __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot ostyp 0 osmod 1 osfin 0 osval 0
    2018-08-09 18:32:14.074 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                          __dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.075 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                  __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.075 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                  __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2 ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.075 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam             __dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface ostyp 0 osmod 1 osfin 3 osval 0.2
    2018-08-09 18:32:14.075 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam               __dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface ostyp 0 osmod 1 osfin 3 osval 0.2
    2018-08-09 18:32:14.075 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam              __dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface ostyp 0 osmod 1 osfin 3 osval 0.2
    2018-08-09 18:32:14.076 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam              __dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.076 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                           __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.076 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                     __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.076 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam                __dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface ostyp 0 osmod 1 osfin 3 osval 1
    2018-08-09 18:32:14.076 ERROR [11499346] [AssimpGGeo::convertMaterials@443]  osnam               __dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface ostyp 0 osmod 1 osfin 3 osval 1
    ...



G4DAE : has ascii fractions for value 
----------------------------------------

::

    epsilon:tmp blyth$ grep opticalsurface g4_00.dae | grep value 
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" type="0" value="1">
          <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" type="0" value="0">
          <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" type="0" value="0">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib1Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib2Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__PmtMtRib3Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInIWSTubSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__TablePanelSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SupportRib1Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SupportRib5Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SlopeRib1Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__SlopeRib5Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__ADVertiCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__ShortParCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnInPiperSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearInnOutPiperSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearIWSCurtainSurface" type="0" value="0.2">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInOWSTubSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__UnistrutRib6Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__UnistrutRib7Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib3Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib5Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib4Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib1Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib2Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib8Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib9Surface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__TopShortCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__TopCornerCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__VertiCableTraySurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOutInPiperSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOutOutPiperSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface" type="0" value="0.2">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__PoolSurfacesAll__LegInDeadTubSurface" type="0" value="1">
          <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" type="0" value="0.2">
    epsilon:tmp blyth$ 



