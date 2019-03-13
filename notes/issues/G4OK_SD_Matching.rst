G4OK_SD_Matching
=================


Geant4
--------

Approach to optical Detection is based on EFFICIENCY value, only 
once the dice falls the right way for detection (as a fraction of absorption)
is the SD associated to LV looked for, and if present a hit is formed.


::

    g4-cls G4OpBoundaryProcess

    G4OpBoundaryProcess.hh fraction of DoAbsorption get `theStatus = Detection` depending
    on EFFICIENCY::

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





     ...
    0539 
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }


    1341 G4bool G4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
    1342 {
    1343   G4Step aStep = *pStep;
    1344 
    1345   aStep.AddTotalEnergyDeposit(thePhotonMomentum);
    1346 
    1347   G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
    1348   if (sd) return sd->Hit(&aStep);
    1349   else return false;
    1350 }



Opticks
----------

Looks like material properties of the sensor are irrelevant currently, 
only the surface properties are relevant (see oxrap/cu/propagate.h) 
with 4 possibilities, with probabilities depending on the surface props:

1. SURFACE_ABSORB
2. SURFACE_DETECT
3. SURFACE_DREFLECT diffuse
4. SURFACE_SREFLECT specular  


Currently when a surface is associated there is no possibility of transmission, 
that only happens on a boundary, generate.cu::

    554 
    555         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    556         {
    557             command = propagate_at_surface(p, s, rng);
    558             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    559             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    560         }
    561         else
    562         {
    563             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    564             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    565             // tacit CONTINUE
    566         }
    567 
    568     }   // bounce < bounce_max

::

    605 __device__ int
    606 propagate_at_surface(Photon &p, State &s, curandState &rng)
    607 {
    608     float u_surface = curand_uniform(&rng);
    609 #ifdef WITH_ALIGN_DEV
    610     float u_surface_burn = curand_uniform(&rng);
    611 #endif
    612 #ifdef WITH_ALIGN_DEV_DEBUG
    613     rtPrintf("propagate_at_surface   u_OpBoundary_DiDiReflectOrTransmit:        %.9g \n", u_surface);
    614     rtPrintf("propagate_at_surface   u_OpBoundary_DoAbsorption:   %.9g \n", u_surface_burn);
    615 #endif
    616 
    617     if( u_surface < s.surface.y )   // absorb   
    618     {
    619         s.flag = SURFACE_ABSORB ;
    620         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    621         return BREAK ;
    622     }
    623     else if ( u_surface < s.surface.y + s.surface.x )  // absorb + detect
    624     {
    625         s.flag = SURFACE_DETECT ;
    626         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    627         return BREAK ;
    628     }
    629     else if (u_surface  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    630     {
    631         s.flag = SURFACE_DREFLECT ;
    632         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    633         return CONTINUE;
    634     }
    635     else
    636     {
    637         s.flag = SURFACE_SREFLECT ;
    638         //propagate_at_specular_reflector(p, s, rng );
    639         propagate_at_specular_reflector_geant4_style(p, s, rng );
    640         return CONTINUE;
    641     }
    642 }


For "isSensor()" surfaces which must have an EFFICIENCY property, currently the probabilities 
are split between only detect/absorb (no reflection off cathode yet).::

     506 GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>* src)
     507 {
     508     GProperty<float>* _detect           = NULL ;
     509     GProperty<float>* _absorb           = NULL ;
     510     GProperty<float>* _reflect_specular = NULL ;
     511     GProperty<float>* _reflect_diffuse  = NULL ;
     512 
     513     if(!src)
     514     {
     515         _detect           = getDefaultProperty(detect);
     516         _absorb           = getDefaultProperty(absorb);
     517         _reflect_specular = getDefaultProperty(reflect_specular);
     518         _reflect_diffuse  = getDefaultProperty(reflect_diffuse);
     519     }
     520     else
     521     {
     522         assert( getStandardDomain() );
     523         assert( src->getStandardDomain() );
     524         
     525         assert(src->isSurface());
     526         GOpticalSurface* os = src->getOpticalSurface() ;  // GSkinSurface and GBorderSurface ctor plant the OpticalSurface into the PropertyMap
     527         assert( os && " all surfaces must have associated OpticalSurface " );
     528         
     529         if(src->isSensor())
     530         {
     531             GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY);
     532             assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );
     533             
     534             if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
     535             {
     536                 _detect           = makeConstantProperty(m_fake_efficiency) ;
     537                 _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
     538                 _reflect_specular = makeConstantProperty(0.0);
     539                 _reflect_diffuse  = makeConstantProperty(0.0);
     540             }   
     541             else
     542             {
     543                 _detect = _EFFICIENCY ;
     544                 _absorb = GProperty<float>::make_one_minus( _detect );
     545                 _reflect_specular = makeConstantProperty(0.0);
     546                 _reflect_diffuse  = makeConstantProperty(0.0);
     547             }   
     548         }
     549         else



So next question : How to get Opticks isSensor assigned for CerenkovMinimal ?
-------------------------------------------------------------------------------

* hmm, vague recollect doing something like this before ... adding pseudo surfaces on Opticks side
  to keep the models aligned : that was before the direct approach 

  * what about :doc:`direct_route_needs_AssimpGGeo_convertSensors_equivalent`

* need to review the direct conversion : g4ok-cd x4-cd


