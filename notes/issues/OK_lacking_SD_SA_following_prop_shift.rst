OK_lacking_SD_SA_following_prop_shift
========================================

FIXED by changed in X4PhysicalVolume to work with GPropertyMap<double>
---------------------------------------------------------------------------

* also GBndLibTest showing the expected surfaces

::

    In [1]: ab.his                                                                                                                                                                                    
    Out[1]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11684     11684       163.44/57 =  2.87  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               42      1724      1721      3              0.00         1.002 +- 0.024        0.998 +- 0.024  [2 ] SI AB
    0001            7ccc2      1562      1406    156              8.20         1.111 +- 0.028        0.900 +- 0.024  [5 ] SI BT BT BT SD
    0002           7ccc62       761       666     95              6.32         1.143 +- 0.041        0.875 +- 0.034  [6 ] SI SC BT BT BT SD
    0003            8ccc2       670       597     73              4.21         1.122 +- 0.043        0.891 +- 0.036  [5 ] SI BT BT BT SA
    0004             8cc2       652       615     37              1.08         1.060 +- 0.042        0.943 +- 0.038  [4 ] SI BT BT SA
    0005              452       396       536   -140             21.03         0.739 +- 0.037        1.354 +- 0.058  [3 ] SI RE AB
    0006              462       447       405     42              2.07         1.104 +- 0.052        0.906 +- 0.045  [3 ] SI SC AB
    0007           7ccc52       378       438    -60              4.41         0.863 +- 0.044        1.159 +- 0.055  [6 ] SI RE BT BT BT SD
    0008           8ccc62       286       262     24              1.05         1.092 +- 0.065        0.916 +- 0.057  [6 ] SI SC BT BT BT SA
    0009          7ccc662       281       222     59              6.92         1.266 +- 0.076        0.790 +- 0.053  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       228       212     16              0.58         1.075 +- 0.071        0.930 +- 0.064  [5 ] SI SC BT BT SA
    0011          7ccc652       189       205    -16              0.65         0.922 +- 0.067        1.085 +- 0.076  [7 ] SI RE SC BT BT BT SD
    0012           8ccc52       188       201    -13              0.43         0.935 +- 0.068        1.069 +- 0.075  [6 ] SI RE BT BT BT SA
    0013            8cc52       154       192    -38              4.17         0.802 +- 0.065        1.247 +- 0.090  [5 ] SI RE BT BT SA
    0014               41       162       145     17              0.94         1.117 +- 0.088        0.895 +- 0.074  [2 ] CK AB
    0015             4552       108       165    -57             11.90         0.655 +- 0.063        1.528 +- 0.119  [4 ] SI RE RE AB
    0016          7ccc552       100       160    -60             13.85         0.625 +- 0.062        1.600 +- 0.126  [7 ] SI RE RE BT BT BT SD
    0017             4cc2       137       115     22              1.92         1.191 +- 0.102        0.839 +- 0.078  [4 ] SI BT BT AB
    0018             4662       139       110     29              3.38         1.264 +- 0.107        0.791 +- 0.075  [4 ] SI SC SC AB
    .                              11684     11684       163.44/57 =  2.87  (pval:0.000 prob:1.000)  



Surfaces or Bnd messed up ?
---------------------------------------------

tds3gun.sh 1::

    In [1]: ab.his                                                                                                                                                                                    
    Out[1]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11684     11684     11470.18/94 = 122.02  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               42      1724      1721      3              0.00         1.002 +- 0.024        0.998 +- 0.024  [2 ] SI AB
    0001            7ccc2         0      1406   -1406           1406.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] SI BT BT BT SD
    0002              452       396       536   -140             21.03         0.739 +- 0.037        1.354 +- 0.058  [3 ] SI RE AB
    0003       ccccccccc2       932         0    932            932.00         0.000 +- 0.000        0.000 +- 0.000  [10] SI BT BT BT BT BT BT BT BT BT
    0004              462       447       405     42              2.07         1.104 +- 0.052        0.906 +- 0.045  [3 ] SI SC AB
    0005           7ccc62         0       666   -666            666.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] SI SC BT BT BT SD
    0006             8cc2         0       615   -615            615.00         0.000 +- 0.000        0.000 +- 0.000  [4 ] SI BT BT SA
    0007            8ccc2         0       597   -597            597.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] SI BT BT BT SA
    0008           7ccc52         0       438   -438            438.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] SI RE BT BT BT SD
    0009       cccccccc62       399         0    399            399.00         0.000 +- 0.000        0.000 +- 0.000  [10] SI SC BT BT BT BT BT BT BT BT
    0010         4cccccc2       347         0    347            347.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] SI BT BT BT BT BT BT AB
    0011               41       162       145     17              0.94         1.117 +- 0.088        0.895 +- 0.074  [2 ] CK AB
    0012             4552       108       165    -57             11.90         0.655 +- 0.063        1.528 +- 0.119  [4 ] SI RE RE AB
    0013           8ccc62         0       262   -262            262.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] SI SC BT BT BT SA
    0014             4cc2       137       115     22              1.92         1.191 +- 0.102        0.839 +- 0.078  [4 ] SI BT BT AB
    0015             4662       139       110     29              3.38         1.264 +- 0.107        0.791 +- 0.075  [4 ] SI SC SC AB
    0016       cccccccc52       246         0    246            246.00         0.000 +- 0.000        0.000 +- 0.000  [10] SI RE BT BT BT BT BT BT BT BT
    0017             4652       110       117     -7              0.22         0.940 +- 0.090        1.064 +- 0.098  [4 ] SI RE SC AB
    0018       cccccbccc2       224         0    224            224.00         0.000 +- 0.000        0.000 +- 0.000  [10] SI BT BT BT BR BT BT BT BT BT
    .                              11684     11684     11470.18/94 = 122.02  (pval:0.000 prob:1.000)  

    In [2]:                


* GSurfaceLibTest properties dump looks OK
* GBndLibTest SMOKING GUN there are no osur/isur


GBndLibTest shows the boundaries lack surfaces
-------------------------------------------------

::

    epsilon:issues blyth$ GBndLibTest 
    2021-06-28 13:00:14.362 INFO  [18138441] [main@113] GBndLibTest
    2021-06-28 13:00:14.368 INFO  [18138441] [main@118]  ok 
    2021-06-28 13:00:14.368 INFO  [18138441] [main@122]  loaded blib 
    2021-06-28 13:00:14.370 INFO  [18138441] [main@126]  loaded all  blib 0x7ff5b36229d0 mlib 0x7ff5b36230e0 slib 0x7ff5b366c630
    2021-06-28 13:00:14.372 INFO  [18138441] [GBndLib::dump@1181] GBndLib::dump
    2021-06-28 13:00:14.372 INFO  [18138441] [GBndLib::dump@1183] GBndLib::dump ni 28
     (  0) om:                 Galactic os:                                is:                                im:                 Galactic (  0)      (16,-1,-1,16)
     (  1) om:                 Galactic os:                                is:                                im:                     Rock (  1)      (16,-1,-1, 8)
     (  2) om:                     Rock os:                                is:                                im:                      Air (  2)      ( 8,-1,-1, 3)
     (  3) om:                      Air os:                                is:                                im:                      Air (  3)      ( 3,-1,-1, 3)
     (  4) om:                      Air os:                                is:                                im:                       LS (  4)      ( 3,-1,-1, 0)
     (  5) om:                      Air os:                                is:                                im:                    Steel (  5)      ( 3,-1,-1, 1)
     (  6) om:                      Air os:                                is:                                im:                    Tyvek (  6)      ( 3,-1,-1, 2)
     (  7) om:                      Air os:                                is:                                im:                Aluminium (  7)      ( 3,-1,-1, 7)
     (  8) om:                Aluminium os:                                is:                                im:                 Adhesive (  8)      ( 7,-1,-1, 6)
     (  9) om:                 Adhesive os:                                is:                                im:              TiO2Coating (  9)      ( 6,-1,-1, 5)
     ( 10) om:              TiO2Coating os:                                is:                                im:             Scintillator ( 10)      ( 5,-1,-1, 4)
     ( 11) om:                     Rock os:                                is:                                im:                    Tyvek ( 11)      ( 8,-1,-1, 2)
     ( 12) om:                    Tyvek os:                                is:                                im:                vetoWater ( 12)      ( 2,-1,-1,15)
     ( 13) om:                vetoWater os:                                is:                                im:       LatticedShellSteel ( 13)      (15,-1,-1, 9)
     ( 14) om:                vetoWater os:                                is:                                im:                    Tyvek ( 14)      (15,-1,-1, 2)
     ( 15) om:                    Tyvek os:                                is:                                im:                    Water ( 15)      ( 2,-1,-1,14)
     ( 16) om:                    Water os:                                is:                                im:                  Acrylic ( 16)      (14,-1,-1,10)
     ( 17) om:                  Acrylic os:                                is:                                im:                       LS ( 17)      (10,-1,-1, 0)
     ( 18) om:                       LS os:                                is:                                im:                  Acrylic ( 18)      ( 0,-1,-1,10)
     ( 19) om:                       LS os:                                is:                                im:                    PE_PA ( 19)      ( 0,-1,-1,11)
     ( 20) om:                    Water os:                                is:                                im:                    Steel ( 20)      (14,-1,-1, 1)
     ( 21) om:                    Water os:                                is:                                im:                    PE_PA ( 21)      (14,-1,-1,11)
     ( 22) om:                    Water os:                                is:                                im:                    Pyrex ( 22)      (14,-1,-1,13)
     ( 23) om:                    Pyrex os:                                is:                                im:                    Pyrex ( 23)      (13,-1,-1,13)
     ( 24) om:                    Pyrex os:                                is:                                im:                   Vacuum ( 24)      (13,-1,-1,12)
     ( 25) om:                    Water os:                                is:                                im:                    Water ( 25)      (14,-1,-1,14)
     ( 26) om:                    Water os:                                is:                                im:                       LS ( 26)      (14,-1,-1, 0)
     ( 27) om:                vetoWater os:                                is:                                im:                    Water ( 27)      (15,-1,-1,14)
    2021-06-28 13:00:14.372 INFO  [18138441] [GBndLib::dumpMaterialLineMap@835] GBndLib::dumpMaterialLineMap
    2021-06-28 13:00:14.372 INFO  [18138441] [GBndLib::dumpMaterialLineMap@840] GBndLib::dumpMaterialLineMap m_materialLineMap.size()  17
    2021-06-28 13:00:14.372 INFO  [18138441] [GBndLib::dumpMaterialLineMap@787] GBndLib::dumpMaterialLineMap
    2021-06-28 13:00:14.372 INFO  [18138441] [GBndLib::dumpMaterialLineMap@790]    67                       Acrylic



::

    epsilon:extg4 blyth$ grep addBoundary *.cc
    X4PhysicalVolume.cc:X4PhysicalVolume::addBoundary
    X4PhysicalVolume.cc:unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
    X4PhysicalVolume.cc:         << " addBoundary "
    X4PhysicalVolume.cc:        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    X4PhysicalVolume.cc:        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    X4PhysicalVolume.cc:        boundary = m_blib->addBoundary( omat, osur, isur, imat ); 
    X4PhysicalVolume.cc:    unsigned boundary = addBoundary( pv, pv_p );
    epsilon:extg4 blyth$ 


        1277 
    1278 #else
    1279     // look for a border surface defined between this and the parent volume, in either direction
    1280     bool first_skin_priority = true ;   // controls fallback skin lv order when bordersurface a->b not found 
    1281     const GPropertyMap<float>* const isur_ = findSurfaceOK(  pv  , pv_p, first_skin_priority );
    1282     const GPropertyMap<float>* const osur_ = findSurfaceOK(  pv_p, pv  , first_skin_priority );
    1283 #endif



Recipe for loss of all surfaces::

     658 
     659 GPropertyMap<float>* X4PhysicalVolume::findSurfaceOK(const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_skin_priority ) const
     660 {
     661      GPropertyMap<float>* surf = nullptr ;
     662 
     663      GBorderSurface* bs = findBorderSurfaceOK( a, b );
     664      surf = dynamic_cast<GPropertyMap<float>*>(bs);
     665 
     666      const G4VPhysicalVolume* const first  = first_skin_priority ? a : b ;
     667      const G4VPhysicalVolume* const second = first_skin_priority ? b : a ;
     668 
     669      if(surf == NULL)
     670      {
     671          GSkinSurface* sk = findSkinSurfaceOK( first ? first->GetLogicalVolume() : NULL );
     672          surf = dynamic_cast<GPropertyMap<float>*>(sk);
     673      }
     674 
     675      if(surf == NULL)
     676      {
     677          GSkinSurface* sk = findSkinSurfaceOK( second ? second->GetLogicalVolume() : NULL );
     678          surf = dynamic_cast<GPropertyMap<float>*>(sk);
     679      }
     680      return surf ;
     681 }








    epsilon:extg4 blyth$ opticks-f "GPropertyMap<float>" 

    ./assimprap/AssimpGGeo.hh:    void addProperties(GPropertyMap<float>* pmap, aiMaterial* material);
    ./assimprap/AssimpGGeo.hh:    void addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property);
    ./assimprap/AssimpGGeo.cc:void AssimpGGeo::addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property )
    ./assimprap/AssimpGGeo.cc:void AssimpGGeo::addProperties(GPropertyMap<float>* pmap, aiMaterial* material )
    ./assimprap/AssimpGGeo.cc:    GPropertyMap<float>* isurf  = NULL ; 
    ./assimprap/AssimpGGeo.cc:    GPropertyMap<float>* osurf  = NULL ; 
    ./assimprap/tests/AssimpGGeoTest.cc:    GPropertyMap<float>*  m_sensor_surface = m_slib->getSensorSurface(0) ;

           assimp is no longer of concern


    ./extg4/X4PhysicalVolume.cc:GPropertyMap<float>* X4PhysicalVolume::findSurfaceOK(const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_skin_priority ) const 
    ./extg4/X4PhysicalVolume.cc:     GPropertyMap<float>* surf = nullptr ; 
    ./extg4/X4PhysicalVolume.cc:     surf = dynamic_cast<GPropertyMap<float>*>(bs); 
    ./extg4/X4PhysicalVolume.cc:         surf = dynamic_cast<GPropertyMap<float>*>(sk); 
    ./extg4/X4PhysicalVolume.cc:         surf = dynamic_cast<GPropertyMap<float>*>(sk); 
    ./extg4/X4PhysicalVolume.cc:    const GPropertyMap<float>* const isur2_ = findSurfaceOK(  pv  , pv_p, first_skin_priority ); 
    ./extg4/X4PhysicalVolume.cc:    const GPropertyMap<float>* const osur2_ = findSurfaceOK(  pv_p, pv  , first_skin_priority ); 
    ./extg4/X4PhysicalVolume.cc:    const GPropertyMap<float>* const isur_ = findSurfaceOK(  pv  , pv_p, first_skin_priority ); 
    ./extg4/X4PhysicalVolume.cc:    const GPropertyMap<float>* const osur_ = findSurfaceOK(  pv_p, pv  , first_skin_priority ); 
    ./extg4/X4PhysicalVolume.hh:        GPropertyMap<float>* findSurfaceOK(const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_skin_priority ) const ; 
    ./extg4/X4MaterialPropertiesTable.hh:GPropertyMap<float> base of GMaterial, GSkinSurface or GBorderSurface.

             fixed all these

    ./ggeo/GMaterial.hh:1. thin layer over base GPropertyMap<float> 
    ./ggeo/GMaterial.hh:2. populated by AssimpGGeo::addProperties(GPropertyMap<float>* pmap, aiMaterial* material )
    ./ggeo/tests/GPropertyMapBaseTest.cc:    GPropertyMap<float>* pmap = new GPropertyMap<float>(matname);
    ./ggeo/tests/GPropertyMapBaseTest.cc:    GPropertyMap<float>* qmap = GPropertyMap<float>::load(matdir, matname, "material");

             fixed all these
               

    ./ggeo/tests/GSurfaceLibTest.cc:    GPropertyMap<float>* m_sensor_surface = NULL ; 


    ./ggeo/GPropertyMap.cc:template class GPropertyMap<float>;
    ./ggeo/GPropertyMap.cc:template GGEO_API void GPropertyMap<float>::setMetaKV(const char* name, int value);
    ./ggeo/GPropertyMap.cc:template GGEO_API void GPropertyMap<float>::setMetaKV(const char* name, std::string value);
    ./ggeo/GPropertyMap.cc:template GGEO_API int         GPropertyMap<float>::getMetaKV(const char* name, const char* fallback) const ;
    ./ggeo/GPropertyMap.cc:template GGEO_API std::string GPropertyMap<float>::getMetaKV(const char* name, const char* fallback) const ;
    ./ggeo/ggeodev.bash:    vector of GPropertyMap<float> corresponding via subclasses 
    epsilon:opticks blyth$ 


