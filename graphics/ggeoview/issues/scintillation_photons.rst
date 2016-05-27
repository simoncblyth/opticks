Scintillation Photons
======================

::

   ggv-;ggv-g4gun
         # geant4 particle gun simulation within default DYB geometry, loaded from GDML
         # using 

OpNovicePhysicsList not yielding scintillation photons. Why?

::

    ggv-;ggv-g4gun --dbg

    (lldb) b "G4Scintillation::PostStepDoIt(G4Track const&, G4Step const&)" 
    Breakpoint 1: where = libG4processes.dylib`G4Scintillation::PostStepDoIt(G4Track const&, G4Step const&) + 39 at G4Scintillation.cc:194, address = 0x00000001035b7c87
    (lldb) b "G4Scintillation::AtRestDoIt(G4Track const&, G4Step const&)" 
    Breakpoint 2: where = libG4processes.dylib`G4Scintillation::AtRestDoIt(G4Track const&, G4Step const&) + 24 at G4Scintillation.cc:179, address = 0x00000001035b7c48

    (lldb) p TotalEnergyDeposit
    (G4double) $2 = 0.25795194278917472

    (lldb) p *aMaterialPropertiesTable    # only standard 4 + REEMISSIONPROB, no FASTCOMPONENT SLOWCOMPONENT 



* the ancient GDML export lacks material props, so added them in from geocache with CPropLib
* but scintillator props are handled differently, the reemiision buffer is 
  calculated and only that gets persisted in GScintillatorLib

* actually GMaterialLib only persists the material buffer also, but the import 
  reconstructs the individual GMaterial... but they are the standard props
  not the raw ones

* need to persist scintillator raw materials in geocache GScintillatorLib
  


Tracing thru Opticks
----------------------

::

    simon:ggeo blyth$ grep FASTCOMP *.*
    GGeo.cc:    findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 
    GScintillatorLib.cc:"fast_component:FASTCOMPONENT," 

In GGeo (pre-cache) scintillator raw GMaterials are selected::

    948 std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, const char* delim)

And used to create the GScintillatorLib which is persisted in geocache::

     806 void GGeo::prepareScintillatorLib()
     807 {
     808     LOG(info) << "GGeo::prepareScintillatorLib " ;
     809 
     810     findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB");
     811 
     812     unsigned int nscint = getNumScintillatorMaterials() ;
     813 
     814     if(nscint == 0)
     815     {
     816         LOG(warning) << "GGeo::prepareScintillatorLib found no scintillator materials  " ;
     817     }
     818     else
     819     {
     820         GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(getScintillatorMaterial(0));
     821 
     822         GScintillatorLib* sclib = getScintillatorLib() ;
     823 
     824         sclib->add(scint);
     825 
     826         sclib->close();
     827     }
     828 }

But just the GPU ready buffer is persisted, not the scintillator materials.


G4Scintillation 
----------------

::

    simon:ggeo blyth$ ( g4-cd ; grep GetConst source/processes/electromagnetic/xrays/src/G4Scintillation.cc )

          GetConstProperty("SCINTILLATIONYIELD");
          GetConstProperty("RESOLUTIONSCALE");
          GetConstProperty("YIELDRATIO");
          GetConstProperty("FASTTIMECONSTANT");
          GetConstProperty("SLOWTIMECONSTANT");


          when fFiniteRiseTime, ie using SetFiniteRiseTime

          GetConstProperty("FASTSCINTILLATIONRISETIME");
          GetConstProperty("SLOWSCINTILLATIONRISETIME");


    simon:ggeo blyth$ ( g4-cd ; grep GetProperty source/processes/electromagnetic/xrays/src/G4Scintillation.cc )

          aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
          aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");

          when using GetScintillationYieldByParticleType

          GetProperty("PROTONSCINTILLATIONYIELD");
          GetProperty("DEUTERONSCINTILLATIONYIELD");
          GetProperty("TRITONSCINTILLATIONYIELD");
          GetProperty("ALPHASCINTILLATIONYIELD");
          GetProperty("IONSCINTILLATIONYIELD");
          GetProperty("ELECTRONSCINTILLATIONYIELD");


DYB Scintillation props
-------------------------

When passing ALL props thru to the MPT::

    GPropertyMap<T>::make_table vprops 6 cprops 5 dprops 5 eprops 5 fprops 2 gprops 0
                  domain           ABSLENGTH       FASTCOMPONENT            RAYLEIGH      REEMISSIONPROB              RINDEX       SLOWCOMPONENT
                     820             2021.01                   0              500000                   0              1.4781                   0
                     800             3358.37                   0              500000                   0              1.4781                   0
                     780             3910.53         0.000177883              460194                   0             1.47845         0.000177883
                     760             989.154         0.000356675              420184                   0             1.47879         0.000356675
                     740                1877         0.000535467              380175                   0             1.47914         0.000535467
                     720             2573.49         0.000714259              340165                   0             1.47949         0.000714259
                     700             4617.72          0.00089305              300155          0.00541012             1.47984          0.00089305
                     680             6944.95          0.00107184              276473           0.0139933             1.48044          0.00107184
                     660             7315.33          0.00125063              252855           0.0225765             1.48127          0.00125063
                     640             5387.98          0.00142943              229236           0.0311597             1.48209          0.00142943
                     620             14952.8          0.00160822              205618           0.0397428             1.48292          0.00160822
                     600             14692.2          0.00178694              181999            0.048326             1.48375          0.00178694
                     580               21528          0.00245293              152790           0.0569092              1.4846          0.00245293
                     560             27079.1          0.00344385              117807           0.0721226             1.48548          0.00344385
                     540             27572.7          0.00696887             93776.7           0.0890841             1.48664          0.00696887
                     520             26137.4           0.0160663             81100.8            0.106046             1.48844           0.0160663
                     500             33867.4           0.0409958               68425              0.1231             1.49024           0.0409958
                     480             27410.1            0.107103             58710.2            0.135033             1.49198            0.107103
                     460             26623.1            0.266127               52039            0.169473             1.49357            0.266127
                     440             28043.5             0.50977             45367.7            0.222394             1.49517             0.50977
                     420             12864.2            0.991617             36028.8            0.496677             1.49718            0.991617
                     400             72.7607              1.0276             27963.7            0.800442             1.50004              1.0276
                     380             4.13274           0.0534018             23891.6            0.800544             1.50531           0.0534018
                     360             1.91251           0.0132052             19819.4            0.800344             1.51058           0.0132052
                     340            0.460452          0.00548866             15747.2            0.800143             1.51586          0.00548866
                     320            0.394828          0.00560594               11675            0.782488             1.52113          0.00560594
                     300            0.329204          0.00467182             7602.84             0.72143              1.5264          0.00467182
                     280             0.26358          0.00373769              6251.1            0.660371             1.54482          0.00373769
                     260            0.197956          0.00280357             4901.25            0.599313             1.56323          0.00280357
                     240            0.132332          0.00186944             3551.41            0.538254             1.58165          0.00186944
                     220           0.0667075         0.000935314             2201.56            0.477196             1.60006         0.000935314
                     200          0.00108341         1.18733e-06             851.716            0.420069             1.61848         1.18733e-06
                     180               0.001                   0                 850            0.410011             1.52723                   0
                     160               0.001                   0                 850             0.40001             1.79251                   0
                     140               0.001                   0                 850                 0.4             1.66438                   0
                     120               0.001                   0                 850                 0.4              1.4536                   0
                     100               0.001                   0                 850                 0.4              1.4536                   0
                      80               0.001                   0                 850                 0.4              1.4536                   0
                      60               0.001                   0                 850                 0.4              1.4536                   0


Constants when passing all DYB scintillator props::

                            domain         AlphaFASTTIMECONSTANT         AlphaSLOWTIMECONSTANT               AlphaYIELDRATIO              FASTTIMECONSTANT         GammaFASTTIMECONSTANT
                               820                             1                            35                          0.65                          3.64                             7
                                60                             1                            35                          0.65                          3.64                             7
                            domain         GammaSLOWTIMECONSTANT               GammaYIELDRATIO       NeutronFASTTIMECONSTANT       NeutronSLOWTIMECONSTANT             NeutronYIELDRATIO
                               820                            31                         0.805                             1                            34                          0.65
                                60                            31                         0.805                             1                            34                          0.65
                            domain               RESOLUTIONSCALE    ReemissionFASTTIMECONSTANT    ReemissionSLOWTIMECONSTANT          ReemissionYIELDRATIO            SCINTILLATIONYIELD
                               820                             1                           1.5                           1.5                             1                         11522
                                60                             1                           1.5                           1.5                             1                         11522
                            domain              SLOWTIMECONSTANT                    YIELDRATIO
                               820                          12.2                          0.86
                                60                          12.2                          0.86


Selected DYB scintillatot props from operation with standard G4Scintillation ie skipping prefixed (Alpha, Gamma, Neutron, Reemission) versions::

                        domain              FASTTIMECONSTANT               RESOLUTIONSCALE            SCINTILLATIONYIELD              SLOWTIMECONSTANT                    YIELDRATIO
                           820                          3.64                             1                         11522                          12.2                          0.86
                            60                          3.64                             1                         11522                          12.2                          0.86





