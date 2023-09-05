leap_to_new_workflow
=======================

* prior :doc:`CSGFoundry_CreateFromSim_shakedown_now_with_flexible_sn`

* CONCLUDED TOO MUCH EFFORT TO BRING OSUR IMPLICITS TO THE OLD WORKFLOW : LEAPING TO NEW WORKFLOW


LEAP ENCOMPASSES

1. hide old world GGeo etc.. behind WITH_GGEO
2. remove old world packages from om-subs--all
3. arrange G4CXOpticks::setGeometry to skip GGeo going
   instead to SSim/stree and then CSGFoundry::CreateFromSim 


SAME COMMAND FROM PREVIOUS NOW USES NEW WORKFLOW
-------------------------------------------------

::

     NEW    U4Tree             CSGImport
     Geant4 -----> SSim/stree ----->  CSGFoundry 
                         
     ~/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.sh   


NEXT
-----

* peel back to Geant4 A/B comparison was doing previously : with 3inch PMT discrep
  motivating the addition of OSUR implicits to NEW workflow


G4CXTest.sh standalone bi-simulation with G4CXApp::Main
---------------------------------------------------------

::

   ~/opticks/g4cx/tests/G4CXTest.sh                       # workstation : run bi-simulation

   ~/opticks/g4cx/tests/G4CXTest.sh grab                  # laptop : grab from workstation

   PICK=AB MODE=3 ~/opticks/g4cx/tests/G4CXTest.sh ana    # laptop : analysis 
   PICK=A MODE=3 ~/opticks/g4cx/tests/G4CXTest.sh ana     # laptop : analysis 



DR spill out ?  NO: THE ISSUE WAS FIXED ORIENT IN qsim::reflect_diffuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Are the boundaries or surfaces messed up ?

* NO : NOT WHEN LOOK AT THE APPROPRIATE GEOM FOR THE EVENT 

:: 

    In [22]: np.c_[cf.sim.bnd_names]
    Out[22]: 
    array([['Rock///Rock'],
           ['Rock//water_rock_bs/Water'],
           ['Water///Pyrex'],
           ['Pyrex/nnvt_photocathode_mirror_logsurf/nnvt_photocathode_mirror_logsurf/Vacuum'],
           ['Vacuum/nnvt_mcp_edge_opsurface/nnvt_photocathode_mirror_logsurf/Steel'],
           ['Vacuum/nnvt_mcp_plate_opsurface/nnvt_photocathode_mirror_logsurf/Steel'],
           ['Vacuum/nnvt_mcp_tube_opsurface/nnvt_photocathode_mirror_logsurf/Steel'],
           ['Vacuum/nnvt_mcp_opsurface/nnvt_photocathode_mirror_logsurf/Steel']], dtype='<U78')


    In [7]: np.c_[cf.sim.stree.standard.bnd_names]     ## AHHA : THESE MAKE MORE SENSE
    Out[7]: 
    array([['Rock///Rock'],
           ['Rock//water_rock_bs/Water'],
           ['Water///Water'],
           ['Water///AcrylicMask'],
           ['Water/NNVTMaskOpticalSurface//CDReflectorSteel']], dtype='<U46')


Full regenerate FewPMT GEOM folders on workstation and laptop to avoid the confusion by
doing the below on both before an run and a "GEOM get"::

    GEOM base
    mv FewPMT old_FewPMT 

Suspect mis-translation or mis-behaviour of NNVTMaskOpticalSurface::

    epsilon:junosw blyth$ jgr MaskOpticalSurface
    ./Simulation/DetSimV2/PMTSim/src/HamamatsuMaskManager.cc:    new G4LogicalSkinSurface("HamamatsuMaskOpticalSurface", logicMaskTail, mask_optical_surface);
    ./Simulation/DetSimV2/PMTSim/src/NNVTMaskManager.cc:    new G4LogicalSkinSurface("NNVTMaskOpticalSurface", logicMaskTail, mask_optical_surface);
    epsilon:junosw blyth$ 

::

    726 void
    727 NNVTMaskManager::makeMaskTailOpSurface(){
    728     G4OpticalSurface* mask_optical_surface = new G4OpticalSurface("opNNVTMask");
    729     mask_optical_surface->SetMaterialPropertiesTable(Steel->GetMaterialPropertiesTable());
    730     mask_optical_surface->SetModel(unified);
    731     mask_optical_surface->SetType(dielectric_metal);
    732     mask_optical_surface->SetFinish(ground);
    733     mask_optical_surface->SetSigmaAlpha(0.2);
    734 
    735     new G4LogicalSkinSurface("NNVTMaskOpticalSurface", logicMaskTail, mask_optical_surface);
    736 }


* note its skin, not border : so should translate into both way 



HMM: does the OpSurface take precedence over no RINDEX from the Steel properties ? 

g4-cls G4OpBoundaryProcess:

* surface check is between the two RINDEX checks but it depends a bit on the finish 
* in the case of ground the lack of RINDEX should not cause fStopAndKill


PIDX debug the reflect_diffuse::

    N[blyth@localhost tests]$ PIDX=1 ./G4CXTest.sh run 

    ...
    //qsim.propagate idx 1 bounce 0 command 3 flag 0 s.optical.x 27 s.optical.y 2 
    //qsim.propagate.WITH_CUSTOM4 idx 1  BOUNDARY ems 2 lposcost  -0.869 
    //qsim.reflect_diffuse idx 1 : old_mom = np.array([   0.00000,   0.00000,  -1.00000]) 
    //qsim.reflect_diffuse idx 1 : normal0 = np.array([   0.00000,   0.00000,   1.00000]) 
    //qsim.reflect_diffuse idx 1 : p.mom = np.array([   0.16561,   0.10511,  -0.98057])     ## UNEXPECTED DIRECTION AGAINST NORMAL 
    //qsim.reflect_diffuse idx 1 : facet_normal = np.array([   0.84019,   0.53326,   0.09856]) 
    //qsim.propagate_at_surface.DR/SR.CONTINUE idx 1 : flag 256 




     
Possible causes:

* not implemented the SigmaAlpha 
* ellipsoid normal lack of normalization : COULD BE AN ISSUE, BUT NOT HERE AS ITS FLAT BASE
* the largest cause was qsim::reflect_diffuse fixed orient bug : FIXED THAT 



::

    465 __global__ void _QSim_lambertian_direction( qsim* sim, quad* q, unsigned num_quad, qdebug* dbg )
    466 {
    467     unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    468     if (idx >= num_quad ) return;
    469 
    470     curandState rng = sim->rngstate[idx] ;
    471 
    472     sctx ctx = {} ;
    473     ctx.idx = idx ;
    474 
    475     float3* dir = (float3*)&q[idx].f.x ;
    476     const float orient = -1.f ;

    ///    FIXED ORIENT HERE : LOOKS WRONG 

    477 
    478     sim->lambertian_direction( dir, &dbg->normal, orient, rng, ctx );
    479 
    480     q[idx].u.w = idx ;
    481 }
    482 


DONE : check reflect_specular fixed orient : removed orient as does not effect calc, hence confusing and pointless
--------------------------------------------------------------------------------------------------------------------

::

    //QSim_photon_launch sim 0x700c61800 photon 0x700c61a00 num_photon 8 dbg 0x700c60a00 type 33 name reflect_specular 
    2023-09-05 12:42:27.597 INFO  [1763025] [QU::copy_device_to_host_and_free@415] copy 8 sizeof(T) 64 label QSim::photon_launch_generate:num_photon
    //qsim.reflect_specular.head idx 0 : normal0 = np.array([   0.00000,   0.00000,   1.00000]) ; orient =   -1.00000 
    //qsim.reflect_specular.head idx 0 : mom0 = np.array([   0.70711,   0.00000,  -0.70711]) 
    //qsim.reflect_specular.head idx 0 : pol0 = np.array([   0.00000,   1.00000,   0.00000]) 
    //qsim.reflect_specular.tail idx 0 : mom1 = np.array([   0.70711,   0.00000,   0.70711]) ; PdotN =    0.70711 ; EdotN =   -0.00000 
    //qsim.reflect_specular.tail idx 0 : pol1 = np.array([   0.00000,  -1.00000,   0.00000]) 
    2023-09-05 12:42:27.600 INFO  [1763025] [main@721]  qst.rc 0
    === eprd.sh : invoking analysis script generic.py
    setting builtins symbol:a gstem:p0

    //qsim.reflect_specular.head idx 0 : normal0 = np.array([   0.00000,   0.00000,   1.00000]) ; orient =    1.00000 
    //qsim.reflect_specular.head idx 0 : mom0 = np.array([   0.70711,   0.00000,  -0.70711]) 
    //qsim.reflect_specular.head idx 0 : pol0 = np.array([   0.00000,   1.00000,   0.00000]) 
    //qsim.reflect_specular.tail idx 0 : mom1 = np.array([   0.70711,   0.00000,   0.70711]) ; PdotN =   -0.70711 ; EdotN =    0.00000 
    //qsim.reflect_specular.tail idx 0 : pol1 = np.array([   0.00000,  -1.00000,   0.00000]) 






TODO : Implement SigmaAlpha less diffuse reflection not just Lambertian
--------------------------------------------------------------------------



::

     638 G4ThreeVector
     639 G4OpBoundaryProcess::GetFacetNormal(const G4ThreeVector& Momentum,
     640                         const G4ThreeVector&  Normal ) const
     641 {
     642         G4ThreeVector FacetNormal;
     643 
     644         if (theModel == unified || theModel == LUT || theModel== DAVIS) {
     645 
     646            /* This function code alpha to a random value taken from the
     647            distribution p(alpha) = g(alpha; 0, sigma_alpha)*std::sin(alpha),
     648            for alpha > 0 and alpha < 90, where g(alpha; 0, sigma_alpha)
     649            is a gaussian distribution with mean 0 and standard deviation
     650            sigma_alpha.  */
     651 
     652            G4double alpha;
     653 
     654            G4double sigma_alpha = 0.0;
     655            if (OpticalSurface) sigma_alpha = OpticalSurface->GetSigmaAlpha();
     656 
     657            if (sigma_alpha == 0.0) return FacetNormal = Normal;
     658 
     659            G4double f_max = std::min(1.0,4.*sigma_alpha);
     660 
     661            G4double phi, SinAlpha, CosAlpha, SinPhi, CosPhi, unit_x, unit_y, unit_z;
     662            G4ThreeVector tmpNormal;
     663 
     664            do {
     665               do {
     666                  alpha = G4RandGauss::shoot(0.0,sigma_alpha);
     667                  // Loop checking, 13-Aug-2015, Peter Gumplinger
     668               } while (G4UniformRand()*f_max > std::sin(alpha) || alpha >= halfpi );
     669 
     670               phi = G4UniformRand()*twopi;
     671 
     672               SinAlpha = std::sin(alpha);
     673               CosAlpha = std::cos(alpha);
     674               SinPhi = std::sin(phi);
     675               CosPhi = std::cos(phi);
     676 
     677               unit_x = SinAlpha * CosPhi;
     678               unit_y = SinAlpha * SinPhi;
     679               unit_z = CosAlpha;
     680 
     681               FacetNormal.setX(unit_x);
     682               FacetNormal.setY(unit_y);
     683               FacetNormal.setZ(unit_z);
     684 
     685               tmpNormal = Normal;
     686 
     687               FacetNormal.rotateUz(tmpNormal);
     688               // Loop checking, 13-Aug-2015, Peter Gumplinger
     689            } while (Momentum * FacetNormal >= 0.0);
     690     }






::

    g4-hh G4RandGauss
    g4-cls Randomize

::

     50 #include "G4MTRandGamma.hh"
     51 #include "G4MTRandGauss.hh"
     52 #include "G4MTRandGaussQ.hh"
     53 #include "G4MTRandGeneral.hh"
     54 
     55 // NOTE: G4RandStat MT-version is missing, but actually currently
     56 // never used in the G4 source
     57 //
     58 #define G4RandFlat G4MTRandFlat
     59 #define G4RandBit G4MTRandBit
     60 #define G4RandGamma G4MTRandGamma
     61 #define G4RandGauss G4MTRandGaussQ
     62 #define G4RandExponential G4MTRandExponential


g4-cls G4MTRandGaussQ::

     68   // Static methods to shoot random values using the static generator
     69 
     70   static  inline G4double shoot();
     71 
     72   static  inline G4double shoot( G4double mean, G4double stdDev );


::


     46 G4double G4MTRandGaussQ::shoot()
     47 {
     48   CLHEP::HepRandomEngine* anEngine = G4MTHepRandom::getTheEngine();
     49   return transformQuick (anEngine->flat());
     50 }


     57 G4double G4MTRandGaussQ::shoot(G4double mean, G4double stdDev)
     58 {
     59   return shoot()*stdDev + mean;
     60 }



    095 // Since all these are this is static to this compilation unit only, the 
     96 // info is establised a priori and not at each invocation.
     97 
     98 // The main data is of course the gaussQTables table; the rest is all 
     99 // bookkeeping to know what the tables mean.
    100 
    101 #define Table0size   250
    102 #define Table1size  1000
    103 #define TableSize   (Table0size+Table1size)
    104 
    105 #define Table0step  (2.0E-6) 
    106 #define Table1step  (5.0E-4)
    107 
    108 #define Table0scale   (1.0/Table1step)
    109 
    110 #define Table0offset 0
    111 #define Table1offset (Table0size)
    112 
    113   // Here comes the big (5K bytes) table, kept in a file ---
    114 
    115 static const G4float gaussTables [TableSize] = {
    116 #include "gaussQTables.cdat"
    117 }; 
    118 
    119 G4double G4MTRandGaussQ::transformQuick (G4double r)                  
    120 {
    121   G4double sign = +1.0; // We always compute a negative number of 
    122                         // sigmas.  For r > 0 we will multiply by
    123                         // sign = -1 to return a positive number.
    124   if ( r > .5 ) {
    125     r = 1-r;
    126     sign = -1.0;
    127   }
    128 
    129   G4int index;
    130   G4double  dx;
    131 
    132   if ( r >= Table1step ) {
    133     index = G4int((Table1size<<1) * r); // 1 to Table1size
    134     if (index == Table1size) return 0.0;
    135     dx = (Table1size<<1) * r - index;  // fraction of way to next bin
    136     index += Table1offset-1;
    137   } else if ( r > Table0step ) {





::

    epsilon:tests blyth$ g4-
    epsilon:tests blyth$ g4-cd
    epsilon:geant4.10.04.p02 blyth$ find . -name gaussQTables.cdat
    ./source/externals/clhep/include/CLHEP/Random/gaussQTables.cdat
    ./source/global/HEPRandom/src/gaussQTables.cdat
    epsilon:geant4.10.04.p02 blyth$ 
    epsilon:geant4.10.04.p02 blyth$ diff source/externals/clhep/include/CLHEP/Random/gaussQTables.cdat source/global/HEPRandom/src/gaussQTables.cdat
    epsilon:geant4.10.04.p02 blyth$ 

