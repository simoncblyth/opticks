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

  * TODO : make max_gs configurable



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
   /tmp/blyth/opticks/evt/g4live/natural/-1/so.npy 
   /tmp/blyth/opticks/cfg4/CCerenkovGeneratorTest/so.npy

* small deviations at 1e-5 level mostly in wavelength 
* TODO: investigate the cause of the deviations : perfect agreement should be possible here 

same ckm exe, cross sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing photons from genstep 0, 
    /tmp/blyth/opticks/evt/g4live/natural/-1/so.npy 
    /tmp/blyth/opticks/evt/g4live/natural/1/ox.npy

* same level of small deviations, 1e~5 level mostly in wavelength 


WIP : For direct workflow : would be more convenient to save events within the keydir 
---------------------------------------------------------------------------------------

The excessive bookeeping with lots of different paths above, motivated
a resource layout rationalization : moving event files into the keydir : in 
source and other subfolders named after executables.

* :doc:`event_path_review`
* :doc:`ckm_cerenkov_generation_align_small_quantized_deviation_g4_g4` 



 
