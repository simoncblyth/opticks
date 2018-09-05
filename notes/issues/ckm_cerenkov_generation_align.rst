ckm_cerenkov_generation_align
==============================


Best of both worlds validation
--------------------------------

* two executables, each a bi-simulation 

::


     (full Geant4 physics)                          (optical only Geant4)
                          
     CerenkovMinimal                                OKG4Test    
 
     G4 example                                     Opticks 
         -  G4OK (Opticks embedded)                   -  CFG4   (Geant4 embedded)




Alignment Preliminaries : basic rng stream matching 
------------------------------------------------------

getting familiar with RNG consumption for cerenkov gen 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* set the embedded commandline for dumping single photon::

   39 const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0"  ;

Cycle on ckm "executable-0"::

    epsilon:g4ok blyth$ t ckm-run
    ckm-run is a function
    ckm-run () 
    { 
        g4-;
        g4-export;
        CerenkovMinimal
    }

* add debug to oxrap/cu/cerenkovstep.h : make sure are seeing expected RNG from TRngBufTest.py 

pindex 0
~~~~~~~~~~~

::

    2018-09-05 10:06:30.331 ERROR [140022] [OPropagator::launch@175] LAUNCH NOW -
    generate photon_id 0 
    gcp.u0    0.74022 wavelength  191.11748 sampledRI    1.36080 
    gcp.u1    0.43845 u_maxSin2    0.15637 sin2Theta    0.35665 
    gcp.u2      0.51701 phi    3.24849 
    gcp.u3    0.15699 delta    0.05618 NumberOfPhotons   27.56718  
    gcp.u4    0.07137 N    1.97903  
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.46250838 speed:220.30603 
    propagate_to_boundary  u_OpRayleigh:0.227643266   scattering_length(s.material1.z):1000000 scattering_distance:1479975.5 
    propagate_to_boundary  u_OpAbsorption:0.329358488   absorption_length(s.material1.y):1000000 absorption_distance:1110608.5 
    propagate_at_surface   u_OpBoundary_DiDiReflectOrTransmit:        0.144065306 
    propagate_at_surface   u_OpBoundary_DoAbsorption:   0.187799111 
     WITH_ALIGN_DEV_DEBUG psave (127.660744 -35.9956245 90 0.727583647) ( -3, 0, 67305985, 129 ) 
    2018-09-05 10:06:30.334 ERROR [140022] [OPropagator::launch@177] LAUNCH DONE

::

    In [5]: a[0].ravel()[:10]
    Out[5]: array([0.74021935, 0.43845114, 0.51701266, 0.15698862, 0.07136751, 0.46250838, 0.22764327, 0.32935849, 0.14406531, 0.18779911])


::

    2018-09-05 10:38:59.618 ERROR [222920] [OPropagator::launch@175] LAUNCH NOW -
    generate photon_id 0 
    gcp.u0    0.74022 wavelength  191.11748 sampledRI    1.36080 
    gcp.u1    0.43845 u_maxSin2    0.15637 sin2Theta    0.35665 
    gcp.u2      0.51701 phi    3.24849 
    gcp.u3    0.15699 delta    0.05618 NumberOfPhotons   27.56718  
    gcp.u4    0.07137 N    1.97903  
     WITH_ALIGN_DEV_DEBUG psave (0.0539942347 -0.0108201597 -0.00217639492 0.000204547003) ( -69271553, 2, 67305985, 1 ) 
    2018-09-05 10:38:59.620 ERROR [222920] [OPropagator::launch@177] LAUNCH DONE

::

    ckm-ecd0
    cd 1
    
    In [8]: a[0,:3]
    Out[8]: 
    array([[  0.054 ,  -0.0108,  -0.0022,   0.0002],
           [  0.7963,  -0.2246,   0.5617,   1.    ],
           [ -0.571 ,   0.0272,   0.8205, 191.1175]], dtype=float32)]


    In [9]: a[1,:3]
    Out[9]: 
    array([[  0.1281,  -0.0257,  -0.0052,   0.0005],
           [  0.9017,   0.3421,   0.2644,   1.    ],
           [ -0.4275,   0.7971,   0.4266, 409.8474]], dtype=float32)



pindex 1
~~~~~~~~~~~

::

    2018-09-05 10:29:00.627 ERROR [185500] [OPropagator::launch@175] LAUNCH NOW -
    generate photon_id 1 
    gcp.u0    0.92099 wavelength  409.84735 sampledRI    1.35449 
    gcp.u1    0.46036 u_maxSin2    0.16419 sin2Theta    0.35064 
    gcp.u2      0.33346 phi    2.09522 
    gcp.u3    0.37252 delta    0.13331 NumberOfPhotons   27.34347  
    gcp.u4    0.48960 N   13.57674  
    WITH_ALIGN_DEV_DEBUG photon_id:1 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.567270935 speed:218.023056 
    propagate_to_boundary  u_OpRayleigh:0.079905808   scattering_length(s.material1.z):1000000 scattering_distance:2526906.75 
    propagate_to_boundary  u_OpAbsorption:0.233368158   absorption_length(s.material1.y):1000000 absorption_distance:1455138 
    propagate_at_surface   u_OpBoundary_DiDiReflectOrTransmit:        0.509377837 
    propagate_at_surface   u_OpBoundary_DoAbsorption:   0.0889785364 
     WITH_ALIGN_DEV_DEBUG psave (307.030823 116.400421 89.9999924 1.56160676) ( -3, 0, 67305985, 65 ) 
    2018-09-05 10:29:00.630 ERROR [185500] [OPropagator::launch@177] LAUNCH DONE

::

    In [8]: a[1].ravel()[:10]
    Out[8]: array([0.9209938 , 0.46036443, 0.33346406, 0.37252042, 0.48960248, 0.56727093, 0.07990581, 0.23336816, 0.50937784, 0.08897854])


Focus on generation "--bouncemax 0"
--------------------------------------- 

::

    39 const char* G4Opticks::fEmbeddedCommandLine = " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  ; 



CCerenkovGeneratorTest
--------------------------

Line up the same gensteps::

     19 int main(int argc, char** argv)
     20 {
     21     OPTICKS_LOG(argc, argv);
     22 
     23     //const char* def = "/usr/local/opticks/opticksdata/gensteps/dayabay/natural/1.npy" ; 
     24     const char* def = "/tmp/blyth/opticks/evt/g4live/natural/1/gs.npy" ;
     25     



::

    (0     :0   ) 0.740219   :      0x1004fc6f4     + 2516 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    2018-09-05 11:12:37.120 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@203]  pindex 0 gcp.u0 0.740219 sampledEnergy 3.58994e-06 sampledRI 1.51438
    (0     :1   ) 0.438451   :      0x1004fca84     + 3428 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    2018-09-05 11:12:37.120 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@215] gcp.u1 0.438451
    (0     :2   ) 0.517013   :      0x1004fcc29     + 3849 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    2018-09-05 11:12:37.120 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@228] gcp.u2 0.517013
    (0     :3   ) 0.156989   :      0x1004fd032     + 4882 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    2018-09-05 11:12:37.180 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@290] gcp.u3 0.156989
    (0     :4   ) 0.071368   :      0x1004fd22b     + 5387 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    2018-09-05 11:12:37.180 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@302] gcp.u4 0.0713675
    (1     :0   ) 0.920994   :      0x1004fc6f4     + 2516 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    (1     :1   ) 0.460364   :      0x1004fca84     + 3428 CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)



back to step collection
-----------------------

::

    epsilon:CerenkovMinimal blyth$ opticks-find collectCerenkovStep
    ./cfg4/C4Cerenkov1042.cc:        CGenstepCollector::Instance()->collectCerenkovStep(
    ./cfg4/CGenstepCollector.cc:void CGenstepCollector::collectCerenkovStep
    ./cfg4/DsG4Cerenkov.cc:        CGenstepCollector::Instance()->collectCerenkovStep(
    ./cfg4/Cerenkov.cc:        CGenstepCollector::Instance()->collectCerenkovStep(
    ./g4ok/G4Opticks.cc:void G4Opticks::collectCerenkovStep
    ./g4ok/G4Opticks.cc:     m_collector->collectCerenkovStep(
    ./examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:        G4Opticks::GetOpticks()->collectCerenkovStep(
    ./cfg4/CGenstepCollector.hh:         void collectCerenkovStep(
    ./g4ok/G4Opticks.hh:        void collectCerenkovStep(


domain range difference
--------------------------

* G4: domain range from original G4Material feeds into Pmin/Pmax
* OK: standardized domain range used




examples/Geant4/CerenkovMinimal/L4Cerenkov.cc
cfg4/C4Cerenkov1042.cc::

    173 G4VParticleChange*
    174 C4Cerenkov1042::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    175 ... 
    190   const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    191   const G4Material* aMaterial = aTrack.GetMaterial();
    192 
    193   G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
    194   G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    195 
    196   G4ThreeVector x0 = pPreStepPoint->GetPosition();
    197   G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
    198   G4double t0 = pPreStepPoint->GetGlobalTime();
    199 
    200   G4MaterialPropertiesTable* aMaterialPropertiesTable =
    201                                aMaterial->GetMaterialPropertiesTable();
    202   if (!aMaterialPropertiesTable) return pParticleChange;
    203 
    204   G4MaterialPropertyVector* Rindex =
    205                 aMaterialPropertiesTable->GetProperty(kRINDEX);
    ...
    256   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    257   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    258   G4double dp = Pmax - Pmin;
    259 
    260   G4double nMax = Rindex->GetMaxValue();
    261 



::

    114 G4MaterialPropertyVector* DetectorConstruction::MakeWaterRI()
    115 {
    116     using CLHEP::eV ;
    117     G4double photonEnergy[] =
    118                 { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
    119                   2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
    120                   2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
    121                   2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
    122                   2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
    123                   3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
    124                   3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
    125                   3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };




::

    2018-09-05 11:27:06.476 INFO  [244122] [CMaterialLib::convert@120] CMaterialLib::convert : converted 38 ggeo materials to G4 materials 
    2018-09-05 11:27:06.476 INFO  [244122] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@125]  Pmin 2.034e-06 Pmax 4.136e-06 wavelength_min(nm) 299.768 wavelength_max(nm) 609.558 meanVelocity 274.664




::

     13 static __device__ __inline__ float boundary_sample_reciprocal_domain(const float& u)
     14 {
     15     // return wavelength, from uniform sampling of 1/wavelength[::-1] domain
     16     float iw = lerp( boundary_domain_reciprocal.x , boundary_domain_reciprocal.y, u ) ;
     17     return 1.f/iw ;
     18 }





cross executable like-to-like alignment  G4-G4 OK-OK 
-------------------------------------------------------


CerenkovMinimal structure
----------------------------

* setup like a Geant4 example using G4OK to access Opticks 
* this means the Opticks commandline is embedded inside G4Opticks

  * so as soon as have control of ckm the idea to do like-for-like matchinh 


0. Getting G4 and Opticks to start from the same RNG stream
--------------------------------------------------------------

Hmm. Recall that the commandline is internal inside 

::

   CerenkovMinimal 
