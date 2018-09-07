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



Alignments:

1. cross executable like-to-like alignment  

   * G4-G4 : not quite the same code, so this checks the genstep transports the stack correctly and the code accomodates
   * OK-OK : should be trivial as same gensteps+code : but need to check anyhow

2. same executable G4-OK

   * difficult one 



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



DONE : reverse translate from texline to material index in GBndLib::MaterialIndexFromLine
--------------------------------------------------------------------------------------------

* material texline is either an inner or outer material

::

    658 unsigned GBndLib::getLine(unsigned ibnd, unsigned imatsur)
    659 {   
    660     assert(imatsur < NUM_MATSUR);  // NUM_MATSUR canonically 4
    661     return ibnd*NUM_MATSUR + imatsur ;
    662 }


DONE : GBndLib::MaterialIndexFromLine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From line to ibnd/imatsur is easy, then can use the optical buffer to get the original index:

1. assert imatsur is 0 or 3 for imat, omat 
2. lookup the 1-based original indices 

::

    ibnd = line / NUM_MATSUR
    imatsur = line - ibnd*NUM_MATSUR 
    

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

    2018-09-05 11:12:37.120 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@203]  pindex 0 gcp.u0 0.740219 sampledEnergy 3.58994e-06 sampledRI 1.51438
    2018-09-05 11:12:37.120 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@215] gcp.u1 0.438451
    2018-09-05 11:12:37.120 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@228] gcp.u2 0.517013
    2018-09-05 11:12:37.180 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@290] gcp.u3 0.156989
    2018-09-05 11:12:37.180 ERROR [237892] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@302] gcp.u4 0.0713675



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


FIXED : domain range difference
-------------------------------------

* G4: domain range from original G4Material feeds into Pmin/Pmax
* OK: standardized domain range used

how to proceed
~~~~~~~~~~~~~~~

* DONE : reverse translate materialLine to materialIndex
* derive the G4 style propertyvec from Opticks standardized material and compare with source
* use the standardized domain interpolated prop for the aligned comparison 

* DONE : hmm will need to rewrite all properties of all materials : so better to 
  do this in one place : not specific to Cerenkov Generator 

* DONE : added assert in CCerenkovGeneration comparing the Rindex domain
  from the material to that from the genstep

* DID NOT DO THIS WAY : clearest way is to used an OPTICKS_ALIGN switch in ckm- DetectorConstruction 
  that does the G4 material standardization, CMaterialLib should do it 
  and should be invoked via some API in G4Opticks 

   * hmm not so sure CMaterialLib is showing its age, its based off of GMaterialLib from hub 
   * X4MaterialTable is what is doing the direct Geant4 to Opticks GGeo conversion, 
     so perhaps X4PhysicalVolume::convertMaterials which populates GMaterialLib directly 
     from Geant4 materials is better for standardize override

   * can have an option to G4Opticks

   * DONE : added standardization option to G4Opticks::setGeometry



1. DONE : got CMaterialLibTest to work from key 

::

    ckm-mlib () 
    { 
        OPTICKS_KEY=$(ckm-key) CMaterialLibTest --envkey
    }



DONE : replace G4Materials with standardized versions 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* G4Material::theMaterialTable structure doesnt make this easy to do

* thought about placement new to replace the G4Material in same location
  of the std::vector<G4Material*> BUT actually there is no need as the material 
  is fine its just an interpolation of properties onto a standard domain and perhaps addition
  of some default properties is all that is needed 

* so just need to replace the MPT::

    G4Material::SetMaterialPropertiesTable(G4MaterialPropertiesTable* anMPT);

* implemented with::

    X4MaterialLib
    X4PropertyMap
    X4Property  


where the Pmin/Pmax comes from in genstep collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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



where the source domain comes from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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





CONFIRMED + FIXED : smoking gun for domain inconsistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    149     G4double BetaInverse = q4.x ;
    150     G4double Pmin = q4.y ;
    151     G4double Pmax = q4.z ;
    152 
    153     G4double wavelength_min = h_Planck*c_light/Pmax ;
    154     G4double wavelength_max = h_Planck*c_light/Pmin ;
    155 
    156     // HMM POTENTIAL FOR BREAKAGE WHEN THE Pmin/Pmax travelling
    157     // via genstep is no longer correct for the rindex of the material



after standardizing materials are getting more gensteps, more photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* not surprising : tis effectively a material change 
* DONE : convenient dev cycle means need a way to terminate after a max_gs count, did this via fStopAndKill G4Track 

  * TODO: make max_gs configurable
  

FIXED : need to domain flip to get same energy sampling for the same randoms : see boundary_lookup.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-09-06 10:52:56.339 INFO  [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@158]  Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 meanVelocity 274.664
    2018-09-06 10:52:56.339 ERROR [765903] [*CCerenkovGenerator::GetRINDEX@72]  aMaterial 0x10a607830 materialIndex 1 num_material 3 Rindex 0x10a609580 Rindex2 0x10a609580
    2018-09-06 10:52:56.339 ERROR [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@251]  genstep_idx 0 fNumPhotons 221 pindex 0
    2018-09-06 10:52:56.339 ERROR [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@283]  gcp.u0 0.740219 sampledEnergy 1.56887e-05 sampledWavelength 79.0277 sampledRI 1.3608
    2018-09-06 10:52:56.339 ERROR [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@295] gcp.u1 0.438451
    2018-09-06 10:52:56.340 ERROR [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@308] gcp.u2 0.517013
    2018-09-06 10:52:56.399 ERROR [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@370] gcp.u3 0.156989
    2018-09-06 10:52:56.400 ERROR [765903] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@382] gcp.u4 0.0713675

    0.740219*(2.0664e-05-1.512e-06)+1.512e-06

    In [1]: 0.740219*(2.0664e-05-1.512e-06)+1.512e-06
    Out[1]: 1.5688674288e-05

::

    generate photon_id 0 
     wavelength_0   60.00000 wavelength_1  820.00018 
    gcp.u0    0.74022 wavelength  191.11748 sampledRI    1.36080 
    gcp.u1    0.43845 u_maxSin2    0.15637 sin2Theta    0.35665 
    gcp.u2      0.51701 phi    3.24849 
    gcp.u3    0.15699 delta    0.05618 NumberOfPhotons  658.09430  
    gcp.u4    0.07137 N   46.97818  


After domain flipping in boundary_lookup get same wavelength:: 

    2018-09-06 11:53:25.773 ERROR [832700] [OPropagator::launch@175] LAUNCH NOW -
    generate photon_id 0 
     wavelength_0  820.00000 wavelength_1   60.00000 
    gcp.u0    0.74022 wavelength   79.02767 sampledRI    1.36080 
    gcp.u1    0.43845 u_maxSin2    0.15637 sin2Theta    0.35665 
    gcp.u2      0.51701 phi    3.24849 
    gcp.u3    0.15699 delta    0.05618 NumberOfPhotons  658.09430  
    gcp.u4    0.07137 N   46.97818  
     WITH_ALIGN_DEV_DEBUG psave (0.0539942347 -0.0108201597 -0.00217639492 0.000204547003) ( -1036, 2, 67305985, 1 ) 
    2018-09-06 11:53:25.775 ERROR [832700] [OPropagator::launch@177] LAUNCH DONE
    
::

     06 rtTextureSampler<float4, 2>  boundary_texture ;
      7 rtDeclareVariable(float4, boundary_domain, , );
      8 rtDeclareVariable(float4, boundary_domain_reciprocal, , );
      9 rtDeclareVariable(uint4,  boundary_bounds, , );
     10 rtDeclareVariable(uint4,  boundary_texture_dim, , );
     11 
     12 
     13 static __device__ __inline__ float boundary_sample_reciprocal_domain(const float& u)
     14 {
     15     // return wavelength, from uniform sampling of 1/wavelength[::-1] domain
     16     float iw = lerp( boundary_domain_reciprocal.x , boundary_domain_reciprocal.y, u ) ;
     17     return 1.f/iw ;
     18 }
     19 

::

    2018-09-06 11:20:09.169 INFO  [818718] [OBndLib::makeBoundaryTexture@161] OBndLib::makeBoundaryTexture buf 3,4,2,39,4 --->  nx 39 ny 24
    2018-09-06 11:20:09.169 INFO  [818718] [OBndLib::makeBoundaryTexture@220] boundary_domain_reciprocal       rdom  0.017   0.001   0.000   0.000 
     rdom.x 0.016667 rdom.y 0.001220 dom.x 60.000000 dom.y 820.000000 dom.z 20.000000 dom.w 760.000000

::

    In [7]: 1./60.   # rdom.x
    Out[7]: 0.016666666666666666

    In [8]: 1./820.  # rdom.y 
    Out[8]: 0.0012195121951219512


    ## lerp 
    ##         (1-t)*a + t*b   =   a + t*(b - a)    
    ##
    ##            


dir pol : cross executable cross sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ckm--::

    2018-09-06 13:28:46.126 INFO  [875245] [OPropagator::prelaunch@155] 1 : (0;221,1) prelaunch_times vali,comp,prel,lnch  0.0000 0.0000 0.9111 0.0000
    2018-09-06 13:28:46.126 ERROR [875245] [OPropagator::launch@175] LAUNCH NOW -
    generate photon_id 0 
     wavelength_0  820.00000 wavelength_1   60.00000 
    gcp.u0    0.74022 wavelength   79.02767 sampledRI    1.36080 
    gcp.u1    0.43845 u_maxSin2    0.15637 sin2Theta    0.35665 
    gcp.u2      0.51701 phi    3.24849 dir (    0.79632   -0.22456    0.56165 ) pol (   -0.57103    0.02716    0.82048 )  
    gcp.u3    0.15699 delta    0.05618 NumberOfPhotons  658.09430  
    gcp.u4    0.07137 N   46.97818  
     WITH_ALIGN_DEV_DEBUG psave (0.0539942347 -0.0108201597 -0.00217639492 0.000204547003) ( -741367809, 2, 67305985, 1 ) 
    2018-09-06 13:28:46.128 ERROR [875245] [OPropagator::launch@177] LAUNCH DONE


ckm-gentest::

    2018-09-06 13:42:33.902 ERROR [880755] [*CCerenkovGenerator::GetRINDEX@72]  aMaterial 0x10b831560 materialIndex 1 num_material 3 Rindex 0x10b8332b0 Rindex2 0x10b8332b0
    2018-09-06 13:42:33.902 ERROR [880755] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@251]  genstep_idx 0 fNumPhotons 221 pindex 0
    2018-09-06 13:42:33.903 ERROR [880755] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@283]  gcp.u0 0.740219 sampledEnergy 1.56887e-05 sampledWavelength 79.0277 hc/nm 0.00123984 sampledRI 1.3608
    2018-09-06 13:42:33.903 ERROR [880755] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@296] gcp.u1 0.438451
    2018-09-06 13:42:33.962 ERROR [880755] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@360] gcp.u2 0.517013 dir ( 0.796318 -0.22456 0.56165 ) pol ( -0.571033 0.027155 0.820478 )
    2018-09-06 13:42:33.963 ERROR [880755] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@385] gcp.u3 0.156989
    2018-09-06 13:42:33.963 ERROR [880755] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@397] gcp.u4 0.0713675


post
~~~~~~~~


ckm--::

    2018-09-06 13:52:00.474 ERROR [887643] [OPropagator::launch@175] LAUNCH NOW -
    generate photon_id 0 
     wavelength_0  820.00000 wavelength_1   60.00000 
    gcp.u0    0.74022 wavelength   79.02767 sampledRI    1.36080 
    gcp.u1    0.43845 u_maxSin2    0.15637 sin2Theta    0.35665 
    gcp.u2      0.51701 phi    3.24849 dir (    0.79632   -0.22456    0.56165 ) pol (   -0.57103    0.02716    0.82048 )  
    gcp.u3    0.15699 delta    0.05618 NumberOfPhotons  658.09430  
    gcp.u4    0.07137 N   46.97818  
    gcp.post (    0.05399   -0.01082   -0.00218 :    0.00020 )  
     WITH_ALIGN_DEV_DEBUG psave (0.0539942347 -0.0108201597 -0.00217639492 0.000204547003) ( -538970241, 2, 67305985, 1 ) 
    2018-09-06 13:52:00.478 ERROR [887643] [OPropagator::launch@177] LAUNCH DONE


ckm-gentest::

    2018-09-06 13:50:43.523 INFO  [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@158]  Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 meanVelocity 274.664
    2018-09-06 13:50:43.523 ERROR [886026] [*CCerenkovGenerator::GetRINDEX@72]  aMaterial 0x10b941990 materialIndex 1 num_material 3 Rindex 0x10b946270 Rindex2 0x10b946270
    2018-09-06 13:50:43.523 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@251]  genstep_idx 0 fNumPhotons 221 pindex 0
    2018-09-06 13:50:43.523 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@283]  gcp.u0 0.740219 sampledEnergy 1.56887e-05 sampledWavelength 79.0277 hc/nm 0.00123984 sampledRI 1.3608
    2018-09-06 13:50:43.523 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@296] gcp.u1 0.438451
    2018-09-06 13:50:43.583 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@360] gcp.u2 0.517013 dir ( 0.796318 -0.22456 0.56165 ) pol ( -0.571033 0.027155 0.820478 )
    2018-09-06 13:50:43.583 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@385] gcp.u3 0.156989
    2018-09-06 13:50:43.583 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@397] gcp.u4 0.0713675
    2018-09-06 13:50:43.583 ERROR [886026] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@428] gcp.post ( 0.053994 -0.010820 -0.002176 : 0.000205)



Genstep 0 comparisons : see ckm-so
---------------------------------------

Comparing Cerenkov generated photons between:

ckm-- 
    CerenkovMinimal : geant4 example app, with genstep and photon collection
    via embedded Opticks with embedded commandline 
    " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  

    --bouncemax 0 
        means that photons are saved immediately after generation, with no propagation 
   
    --printenabled --pindex 0
        dump kernel debug for photon 0 


ckm-gentest : 
    CCerenkovGeneratorTest : genstep eating standalone CPU generator that tries to
    mimic the cerenkov process photons via verbatim code copy 

    genstep source set in main at: "/tmp/blyth/opticks/evt/g4live/natural/1/gs.npy"


cross exe, "same" sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing photons from genstep 0, 
   /tmp/blyth/opticks/evt/g4live/natural/~1/so.npy 
   /tmp/blyth/opticks/cfg4/CCerenkovGeneratorTest/so.npy

* small deviations at 1e~5 level mostly in wavelength 

same ckm exe, cross sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing photons from genstep 0, 
    /tmp/blyth/opticks/evt/g4live/natural/~1/so.npy 
    /tmp/blyth/opticks/evt/g4live/natural/1/ox.npy

* same level of small deviations, 1e~5 level mostly in wavelength 


cross exe, same sim
~~~~~~~~~~~~~~~~~~~~~


WIP : For direct workflow : would be more convenient to save events within the keydir 
------------------------------------------------------------------------------------------

* :doc:`event_path_review`

* especially important for passing gensteps between executables, for direct running
  can have a standard path within the keydir for each tag at which to look for gensteps

  * first geometry + genstep collecting and writing executable is special : it should write its event
    and genstep into distinctive "standard" directory (perhaps under the name "source") within the
    geocache keydir 

  * all other executables sharing the same keydir can put their events underneath 
    a relpath named after the executable  
   

* context :doc:`ckm_cerenkov_generation_align`


former non-scalable approach
-----------------------------

* non-scalable because there is no tag or typ specification just the gensteps directly in geocache 

* HMM : is anything more than tag actually needed in direct route ?


::

    epsilon:source blyth$ opticks-find directgensteppath 
    ./boostrap/BOpticksResource.cc:    m_directgensteppath(NULL),
    ./boostrap/BOpticksResource.cc:    m_directgensteppath = makeIdPathPath("directgenstep.npy");  
    ./boostrap/BOpticksResource.cc:    m_res->addPath("directgensteppath", m_directgensteppath ); 
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getDirectGenstepPath() const { return m_directgensteppath ; } 
    ./boostrap/BOpticksResource.hh:       const char* m_directgensteppath ; 
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ opticks-find getDirectGenstepPath
    ./g4ok/G4Opticks.cc:    const char* gspath = m_ok->getDirectGenstepPath(); 
    ./optickscore/Opticks.cc:const char* Opticks::getDirectGenstepPath() const { return m_resource->getDirectGenstepPath() ; } 
    ./optickscore/Opticks.cc:    const char* path = getDirectGenstepPath();
    ./optickscore/Opticks.cc:    std::string path = getDirectGenstepPath();
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getDirectGenstepPath() const { return m_directgensteppath ; } 
    ./optickscore/Opticks.hh:       const char*          getDirectGenstepPath() const ; 
    ./boostrap/BOpticksResource.hh:       const char* getDirectGenstepPath() const ;
    epsilon:opticks blyth$ 

::

    207 /**
    208 G4Opticks::propagateOpticalPhotons
    209 -----------------------------------
    210 
    211 Invoked from EventAction::EndOfEventAction
    212 
    213 TODO: relocate direct events inside the geocache ? 
    214       and place these direct gensteps and genphotons 
    215       within the OpticksEvent directory 
    216 
    217 
    218 **/
    219 
    220 int G4Opticks::propagateOpticalPhotons()
    221 {
    222     m_gensteps = m_genstep_collector->getGensteps();
    223     const char* gspath = m_ok->getDirectGenstepPath();
    224 
    225     LOG(info) << " saving gensteps to " << gspath ;
    226     m_gensteps->setArrayContentVersion(G4VERSION_NUMBER);
    227     m_gensteps->save(gspath);
    228 

    1898 bool Opticks::existsDirectGenstepPath() const
    1899 {
    1900     const char* path = getDirectGenstepPath();
    1901     return path ? BFile::ExistsFile(path) : false ;
    1902 }
    1903 

    epsilon:optickscore blyth$ opticks-find loadDirectGenstep
    ./opticksgeo/OpticksGen.cc:    m_direct_gensteps(m_ok->existsDirectGenstepPath() ? m_ok->loadDirectGenstep() : NULL ),
    ./optickscore/Opticks.cc:NPY<float>* Opticks::loadDirectGenstep() const 
    ./optickscore/Opticks.hh:       NPY<float>*          loadDirectGenstep() const ;
    epsilon:opticks blyth$ 



OpticksGen auto sets the sourcecode based on existance of direct gensteps::

     28 OpticksGen::OpticksGen(OpticksHub* hub)
     29     :
     30     m_hub(hub),
     31     m_gun(new OpticksGun(hub)),
     32     m_ok(hub->getOpticks()),
     33     m_cfg(m_ok->getCfg()),
     34     m_ggb(hub->getGGeoBase()),
     35     m_blib(m_ggb->getBndLib()),
     36     m_lookup(hub->getLookup()),
     37     m_torchstep(NULL),
     38     m_fabstep(NULL),
     39     m_input_gensteps(NULL),
     40     m_csg_emit(hub->findEmitter()),
     41     m_emitter_dbg(false),
     42     m_emitter(m_csg_emit ? new NEmitPhotonsNPY(m_csg_emit, EMITSOURCE, m_ok->getSeed(), m_emitter_dbg, m_ok->getMaskBuffer()) : NULL ),
     43     m_input_photons(NULL),
     44     m_input_primaries(m_ok->existsPrimariesPath() ? m_ok->loadPrimaries() : NULL ),
     45     m_direct_gensteps(m_ok->existsDirectGenstepPath() ? m_ok->loadDirectGenstep() : NULL ),
     46     m_source_code(initSourceCode())
     47 {
     48     init() ;
     49 }
     50 
     51 Opticks* OpticksGen::getOpticks() const { return m_ok ; }
     52 std::string OpticksGen::getG4GunConfig() const { return m_gun->getConfig() ; }
     53 
     54 bool OpticksGen::hasInputPrimaries() const
     55 {
     56     return m_input_primaries != NULL ;
     57 }
     58 
     59 
     60 unsigned OpticksGen::initSourceCode() const
     61 {
     62     unsigned code = 0 ;
     63     if(m_direct_gensteps)
     64     {
     65         code = GENSTEPSOURCE ;
     66     } 
     67     else if(m_input_primaries)
     68     {
     69         code = PRIMARYSOURCE ;
     70     } 
     71     else if(m_emitter)





How to distinguish the special key creating executable from key reading ?
----------------------------------------------------------------------------

The distinguishing thing is the direct translation of geometry done in G4Opticks::translateGeometry
so perhaps a "keysource" flag option for  BOpticksKey::SetKey(keyspec)

* actually can auto-detect this from the exename of the key matching that of the current executable

::

    139 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    140 {
    141     const char* keyspec = X4PhysicalVolume::Key(top) ;
    142     BOpticksKey::SetKey(keyspec);
    143     LOG(error) << " SetKey " << keyspec  ;
    144 
    145     Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    146 


How to allow other executables to access paths written by the keysource executable ?
---------------------------------------------------------------------------------------

Provide two dirs, which are the same for the KeySource case, so can always write to evtbase
and can read from srcevtbase.

::

    474     const char* user = SSys::username();
    475     m_srcevtbase = makeIdPathPath("evt", user, "source");
    476     m_res->addDir( "srcevtbase", m_srcevtbase );
    477 
    478     const char* exename = SAr::Instance->exename();
    479     m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("evt", user, exename ) ;
    480     m_res->addDir( "evtbase", m_evtbase );


How to handle multiple users sharing a geocache ?
---------------------------------------------------

* could move current TMP /tmp/username/opticks into  keydir ?

  * using a username dir  


Event Path machinery 
----------------------

NPY has special handling of quad-argument save::

     707 template <typename T>
     708 NPY<T>* NPY<T>::load(const char* tfmt, const char* source, const char* tag, const char* det, bool quietly)
     709 {
     710     //  (ox,cerenkov,1,dayabay)  ->   (dayabay,cerenkov,1,ox)
     711     //
     712     //     arg order twiddling done here is transitional to ease the migration 
     713     //     once working in the close to old arg order, can untwiddling all the calls
     714     //
     715     std::string path = NPYBase::path(det, source, tag, tfmt );
     716     return load(path.c_str(),quietly);
     717 }
     718 template <typename T>
     719 void NPY<T>::save(const char* tfmt, const char* source, const char* tag, const char* det)
     720 {
     721     std::string path_ = NPYBase::path(det, source, tag, tfmt );
     722     save(path_.c_str());
     723 }
     724 
     
::

    102 std::string NPYBase::path(const char* dir, const char* reldir, const char* name)
    103 {
    104     std::string path = BOpticksEvent::path(dir, reldir, name);
    105     return path ;
    106 }
    107 
    108 std::string NPYBase::path(const char* dir, const char* name)
    109 {
    110     std::string path = BOpticksEvent::path(dir, name);
    111     return path ;
    112 }
    113 
    114 std::string NPYBase::path(const char* det, const char* source, const char* tag, const char* tfmt)
    115 {
    116     std::string path = BOpticksEvent::path(det, source, tag, tfmt );
    117     return path ;
    118 }



::

     14 const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
     15 const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
     16 const char* BOpticksEvent::OVERRIDE_EVENT_BASE = NULL ;
     17 
     18 const int BOpticksEvent::DEFAULT_LAYOUT_VERSION = 2 ;
     19 int BOpticksEvent::LAYOUT_VERSION = 2 ;
     20 

::

    epsilon:boostrap blyth$ opticks-find OPTICKS_EVENT_BASE
    ./boostrap/BFile.cc:           else if(evalue.compare("OPTICKS_EVENT_BASE")==0) 
    ./boostrap/BFile.cc:               LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ; 
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:       LOG(debug) << "BOpticksEvent::directory_template OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
    ./boostrap/BOpticksEvent.cc:       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    ./ana/ncensus.py:    c = Census("$OPTICKS_EVENT_BASE/evt")
    ./ana/nload.py:DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
    epsilon:opticks blyth$ 


BFile.cc OPTICKS_EVENT_BASE is not an envvar but it is internally treated a bit like one, which works
as all file access goes thru BFile::FormPath::

    087 std::string expandvar(const char* s)
     88 {
     89     fs::path p ;
     90 
     91     std::string dollar("$");
     92     boost::regex e("(\\$)(\\w+)(.*?)"); // eg $HOME/.opticks/hello
     93     boost::cmatch m ;
     94 
     95     if(boost::regex_match(s,m,e))
     96     {
     97         //dump(m);  
     98 
     99         unsigned int size = m.size();
    100 
    101         if(size == 4 && dollar.compare(m[1]) == 0)
    102         {
    103            std::string key = m[2] ;
    104 
    105            const char* evalue_ = SSys::getenvvar(key.c_str()) ;
    106 
    107            std::string evalue = evalue_ ? evalue_ : key ;
    108 
    109            if(evalue.compare("TMP")==0) //  TMP envvar not defined
    110            {
    111                evalue = usertmpdir("/tmp","opticks", NULL);
    112                LOG(verbose) << "expandvar replacing TMP with " << evalue ;
    113            }
    114            else if(evalue.compare("TMPTEST")==0)
    115            {
    116                evalue = usertmpdir("/tmp","opticks","test");
    117                LOG(verbose) << "expandvar replacing TMPTEST with " << evalue ;
    118            }
    119            else if(evalue.compare("OPTICKS_EVENT_BASE")==0)
    120            {
    121                evalue = usertmpdir("/tmp","opticks",NULL);
    122                LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ;
    123            }
    124 
    125 
    126            p /= evalue ;
    127 
    128            std::string tail = m[3] ;
    129 
    130            p /= tail ;



CerenkovMinimal::

     18 void RunAction::BeginOfRunAction(const G4Run*)
     19 {
     20     LOG(info) << "." ;
     21 #ifdef WITH_OPTICKS
     22     G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
     23     assert( world ) ;
     24     bool standardize_geant4_materials = true ;   // required for alignment 
     25     G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );
     26 #endif
     27 }

Direct route, keyspec required to be set prior to Opticks instanciation::

    139 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
    140 {
    141     const char* keyspec = X4PhysicalVolume::Key(top) ;
    142     BOpticksKey::SetKey(keyspec);
    143     LOG(error) << " SetKey " << keyspec  ;
    144 
    145     Opticks* ok = new Opticks(0,0, fEmbeddedCommandLine);  // Opticks instanciation must be after BOpticksKey::SetKey
    146 


::

     28 BOpticksResource::BOpticksResource()
     29     :
     30     m_log(new SLog("BOpticksResource::BOpticksResource","",debug)),
     31     m_setup(false),
     32     m_key(BOpticksKey::GetKey()),   // will be NULL unless BOpticksKey::SetKey has been called 
     33     m_id(NULL),

::
 
     248 void OpticksResource::init()
     249 {
     250    LOG(LEVEL) << "OpticksResource::init" ;
     251 
     252    BStr::split(m_detector_types, "GScintillatorLib,GMaterialLib,GSurfaceLib,GBndLib,GSourceLib", ',' );
     253    BStr::split(m_resource_types, "GFlags,OpticksColors", ',' );
     254 
     255    readG4Environment();
     256    readOpticksEnvironment();
     257 
     258    if( m_key )
     259    {
     260        setupViaKey();    // from BOpticksResource base
     261    }
     262    else
     263    {
     264        readEnvironment();
     265    }
     266 
     267    readMetadata();
     268    identifyGeometry();
     269    assignDetectorName();
     270    assignDefaultMaterial();
     271 
     272    LOG(LEVEL) << "OpticksResource::init DONE" ;
     273 }




Hmm having username prefix for source would be inconvenient cross user, also evt is duplicated::

    epsilon:1 blyth$ find evt
    evt
    evt/blyth
    evt/blyth/source
    evt/blyth/source/evt
    evt/blyth/source/evt/g4live
    evt/blyth/source/evt/g4live/natural
    evt/blyth/source/evt/g4live/natural/-1
    evt/blyth/source/evt/g4live/natural/-1/ht.npy
    evt/blyth/source/evt/g4live/natural/-1/so.npy
    evt/blyth/source/evt/g4live/natural/-1/so.json
    evt/blyth/source/evt/g4live/natural/Opticks.npy
    evt/blyth/source/evt/g4live/natural/DeltaVM.ini
    evt/blyth/source/evt/g4live/natural/1
    evt/blyth/source/evt/g4live/natural/1/ps.npy
    evt/blyth/source/evt/g4live/natural/1/ht.npy
    evt/blyth/source/evt/g4live/natural/1/rx.npy
    evt/blyth/source/evt/g4live/natural/1/History_SequenceSource.json
    evt/blyth/source/evt/g4live/natural/1/parameters.json
    evt/blyth/source/evt/g4live/natural/1/Material_SequenceLocal.json
    evt/blyth/source/evt/g4live/natural/1/History_SequenceLocal.json
    evt/blyth/source/evt/g4live/natural/1/20180906_190855
    evt/blyth/source/evt/g4live/natural/1/20180906_190855/parameters.json
    evt/blyth/source/evt/g4live/natural/1/20180906_190855/t_delta.ini
    evt/blyth/source/evt/g4live/natural/1/20180906_190855/t_absolute.ini
    evt/blyth/source/evt/g4live/natural/1/20180906_190855/report.txt
    evt/blyth/source/evt/g4live/natural/1/fdom.npy
    evt/blyth/source/evt/g4live/natural/1/Boundary_IndexLocal.json
    evt/blyth/source/evt/g4live/natural/1/t_delta.ini
    evt/blyth/source/evt/g4live/natural/1/ox.npy
    evt/blyth/source/evt/g4live/natural/1/t_absolute.ini
    evt/blyth/source/evt/g4live/natural/1/Boundary_IndexSource.json


