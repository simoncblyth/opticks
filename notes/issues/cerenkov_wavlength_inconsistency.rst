cerenkov_wavlength_inconsistency
===================================

Fix : G4Opticks::collectGenstep_G4Cerenkov_1042
--------------------------------------------------


optixrap/cu/cerenkovstep.h::


    @@ -258,7 +259,10 @@ generate_cerenkov_photon(Photon& p, CerenkovStep& cs, curandState &rng)
         
             u = curand_uniform(&rng) ; 
     
    -        wavelength = boundary_sample_reciprocal_domain_v3(u);  // TODO: sampling range needs to come from the genstep not default domain
    +        //wavelength = boundary_sample_reciprocal_domain_v3(u);  // TODO: sampling range needs to come from the genstep not default domain
    +
    +        wavelength = lerp( cs.Pmin, cs.Pmax, u );  // the Pmin, Pmax names are misleading they are Wmin, Wmax
    +        //rtPrintf(" wavelength %10.4f cs.Pmin %10.4f cs.Pmax %10.4f u %10.4f \n", wavelength, cs.Pmin, cs.Pmax, u ); 
     




::

    1609 CGenstep G4Opticks::collectGenstep_G4Cerenkov_1042(
    1610      const G4Track*  aTrack,
    1611      const G4Step*   aStep,
    1612      G4int       numPhotons,
    1613 
    1614      G4double    betaInverse,
    1615      G4double    pmin,
    1616      G4double    pmax,
    1617      G4double    maxCos,
    1618 
    1619      G4double    maxSin2,
    1620      G4double    meanNumberOfPhotons1,
    1621      G4double    meanNumberOfPhotons2
    1622     )
    1623 {
    1624     G4StepPoint* pPreStepPoint  = aStep->GetPreStepPoint();
    1625     G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();
    1626 
    1627     G4ThreeVector x0 = pPreStepPoint->GetPosition();
    1628     G4double      t0 = pPreStepPoint->GetGlobalTime();
    1629     G4ThreeVector deltaPosition = aStep->GetDeltaPosition() ;
    1630 
    1631     G4double wmin_nm = h_Planck*c_light/pmax/nm ;
    1632     G4double wmax_nm = h_Planck*c_light/pmin/nm ;
    1633     bool wl_minmax = true ;
    1634 
    1635     LOG(LEVEL)
    1636         << " pmin/eV " << std::setw(10) << std::fixed << std::setprecision(3) << pmin/eV
    1637         << " pmax/eV " << std::setw(10) << std::fixed << std::setprecision(3) << pmax/eV
    1638         << " wmin_nm  " << std::setw(10) << std::fixed << std::setprecision(3) << wmin_nm
    1639         << " wmax_nm  " << std::setw(10) << std::fixed << std::setprecision(3) << wmax_nm
    1640         << " wl_minmax " << wl_minmax
    1641         ;
    ....
    1650     CGenstep gs = m_genstep_collector->collectCerenkovStep(
    1651 
    1652          OpticksGenstep_G4Cerenkov_1042,                // (int)gentype       (0)
    1653          aTrack->GetTrackID(),                          // (int)ParenttId     
    1654          aMaterial->GetIndex(),                         // (int)MaterialIndex
    1655          numPhotons,                                    // (int)NumPhotons
    1656 
    1657          x0.x(),                                        // x0.x               (1)
    1658          x0.y(),                                        // x0.y
    1659          x0.z(),                                        // x0.z
    1660          t0,                                            // t0 
    1661 
    1662          deltaPosition.x(),                             // DeltaPosition.x    (2)
    1663          deltaPosition.y(),                             // DeltaPosition.y    
    1664          deltaPosition.z(),                             // DeltaPosition.z    
    1665          aStep->GetStepLength(),                        // step_length 
    1666 
    1667          aParticle->GetDefinition()->GetPDGEncoding(),  // (int)code          (3) 
    1668          aParticle->GetDefinition()->GetPDGCharge(),    // charge
    1669          aTrack->GetWeight(),                           // weight 
    1670          preVelocity,                                   // preVelocity 
    1671 
    1672          betaInverse,                                   //                    (4) 
    1673          wl_minmax ? wmin_nm : pmin,
    1674          wl_minmax ? wmax_nm : pmax,
    1675          maxCos,
    1676 
    1677          maxSin2,                                       //                    (5)
    1678          meanNumberOfPhotons1,
    1679          meanNumberOfPhotons2,
    1680          postVelocity
    1681     );
    1682 
    1683     LOG(LEVEL)  << gs.desc() ; 
    1684     return gs ;
    1685 }
    1686 




Issue
--------

Hi Zike, 

Thanks for the problem report.

When you find something unusual you should examine the 
code and trace the relevant parts of the implementation and try 
to find the bug / bad input / misunderstanding or inconsistency.
As I do below.


In this case you can start from the GPU implementation:

    optixrap/cu/cerenkovstep.h


257      do {
258 
259         u = curand_uniform(&rng) ;
260 
261         wavelength = boundary_sample_reciprocal_domain_v3(u);
262 
263         float4 props = boundary_lookup(wavelength, cs.MaterialIndex, 0);  // USING cs.MaterialIndex not using geometry 
264 
265         sampledRI = props.x ;
266 
267 #ifdef ALIGN_DEBUG
268         rtPrintf("gcp.u0 %10.5f wavelength %10.5f sampledRI %10.5f \n", u, wavelength, sampledRI  );
269 #endif
270 
271         cosTheta = cs.BetaInverse / sampledRI;
272 
273         sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  // avoid going -ve 
274 
275         u = curand_uniform(&rng) ;
276 
277         u_maxSin2 = u*cs.maxSin2 ;
278 
279 #ifdef ALIGN_DEBUG
280         rtPrintf("gcp.u1 %10.5f u_maxSin2 %10.5f sin2Theta %10.5f \n", u, u_maxSin2, sin2Theta  );
281 #endif
282 
283 
284       } while ( u_maxSin2 > sin2Theta);
285 
286       p.wavelength = wavelength ;
287 



Contrast that with the Geant4 implementation


g4-cls G4Cerenkov


250 
251   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
252   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
253   G4double dp = Pmax - Pmin;
254 
255   G4double nMax = Rindex->GetMaxValue();
256 
257   G4double BetaInverse = 1./beta;
258 
259   G4double maxCos = BetaInverse / nMax;
260   G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
261 
262   G4double beta1 = pPreStepPoint ->GetBeta();
263   G4double beta2 = pPostStepPoint->GetBeta();
264 
265   G4double MeanNumberOfPhotons1 =
266                      GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
267   G4double MeanNumberOfPhotons2 =
268                      GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);
269   
270   for (G4int i = 0; i < fNumPhotons; i++) {
271 
272       // Determine photon energy
273   
274       G4double rand;
275       G4double sampledEnergy, sampledRI;
276       G4double cosTheta, sin2Theta;
277 
278       // sample an energy
279   
280       do {
281          rand = G4UniformRand();
282          sampledEnergy = Pmin + rand * dp;
283          sampledRI = Rindex->Value(sampledEnergy);
284          cosTheta = BetaInverse / sampledRI;
285   
286          sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
287          rand = G4UniformRand();
288 
289         // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
290       } while (rand*maxSin2 > sin2Theta);
291   
292       // Generate random position of photon on cone surface 
293       // defined by Theta 
294 




The GPU implementation gets sampled wavelength with:

 56 static __device__ __inline__ float boundary_sample_reciprocal_domain_v3(const float& u)
 57 {
 58     // see boundary_lookup.py
 59     float a = boundary_domain.x ;
 60     float b = boundary_domain.y ;
 61     return a*b/lerp( a, b, u ) ;
 62 }
 63 

The wavelength from here should be between a and b and the uniform random 
sampling is done in a reciprocal way to match uniform linear energy sampling.

The boundary_domain comes from 

    optixrap/OBndLib.cc


231 
232     bool fine = nl == Opticks::FINE_DOMAIN_LENGTH ;
233     glm::vec4 dom = Opticks::getDomainSpec(fine) ;
234     glm::vec4 rdom = Opticks::getDomainReciprocalSpec(fine) ;
235 
236     m_context["boundary_texture"]->setTextureSampler(tex);
237     m_context["boundary_texture_dim"]->setUint(texDim);
238 
239     m_context["boundary_domain"]->setFloat(dom.x, dom.y, dom.z, dom.w);
240     m_context["boundary_domain_reciprocal"]->setFloat(rdom.x, rdom.y, rdom.z, rdom.w);
241     m_context["boundary_bounds"]->setUint(bounds);
242 
243 


  optickscore/Opticks.cc

 138 // formerly of GPropertyLib, now booted upstairs
 139 float        Opticks::DOMAIN_LOW  = 60.f ;
 140 float        Opticks::DOMAIN_HIGH = 820.f ;  // has been 810.f for a long time  
 141 float        Opticks::DOMAIN_STEP = 20.f ;
 142 unsigned     Opticks::DOMAIN_LENGTH = 39  ;


 198 glm::vec4 Opticks::getDomainSpec(bool fine)
 199 {
 200     glm::vec4 bd ;
 201 
 202     bd.x = DOMAIN_LOW ;
 203     bd.y = DOMAIN_HIGH ;
 204     bd.z = fine ? FINE_DOMAIN_STEP : DOMAIN_STEP ;
 205     bd.w = DOMAIN_HIGH - DOMAIN_LOW ;
 206 
 207     return bd ;
 208 }


 
The above explains the wavelength obtained and 
shows an inconsistency with Geant4.

Geant4 is using the energy range based on the Rindex property 
of the material:

 251   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
 252   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();


Opticks is currently using a fixed wavelength domain.
Clearly I need to fix this to follow the Geant4 approach.

However before fixing this I will put together a test 
that compares the Geant4 and Opticks wavelength distributions, 
so I can see the before and after effect of the fix. 

I will let you know when I have made that change.  


Also see the below which is a standalone test of Geant4 Cerenkov
might to interesting for you:

    examples/Geant4/CerenkovMinimal/src/L4CerenkovTest.cc
    examples/Geant4/CerenkovMinimal/src/L4CerenkovTest.sh
    examples/Geant4/CerenkovMinimal/src/L4CerenkovTest.py
    

Simon

