ground_lambertian_reflection
==============================


Below is a brief look at the relevant Geant4 code that implements "groundfrontpainted",
it seems to just switch on LambertianReflection.
It all looks perfectly doable within Opticks. But as I have no need of it
you will have to do the work if you need it.



epsilon:np blyth$ g4-cc groundfrontpainted
/usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/optical/src/G4OpBoundaryProcess.cc:                else if ( theFinish == groundfrontpainted ) {
/usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLReadSolids.cc:   if ((sfinish=="groundfrontpainted") || (sfinish=="4"))
/usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLReadSolids.cc:      { finish = groundfrontpainted; } else
epsilon:np blyth$ 


g4-cls G4OpBoundaryProcess

 492         else if (type == dielectric_dielectric) {
 493 
 494           if ( theFinish == polishedbackpainted ||
 495                theFinish == groundbackpainted ) {
 496              DielectricDielectric();
 497           }
 498           else {
 499              G4double rand = G4UniformRand();
 500              if ( rand > theReflectivity ) {
 501                 if (rand > theReflectivity + theTransmittance) {
 502                    DoAbsorption();
 503                 } else {
 504                    theStatus = Transmission;
 505                    NewMomentum = OldMomentum;
 506                    NewPolarization = OldPolarization;
 507                 }
 508              }
 509              else {
 510                 if ( theFinish == polishedfrontpainted ) {
 511                    DoReflection();
 512                 }
 513                 else if ( theFinish == groundfrontpainted ) {
 514                    theStatus = LambertianReflection;
 515                    DoReflection();
 516                 }
 517                 else {
 518                    DielectricDielectric();
 519                 }
 520              }
 521           }
 522         }
 523         else {


344 inline
345 void G4OpBoundaryProcess::DoReflection()
346 {
347         if ( theStatus == LambertianReflection ) {
348
349           NewMomentum = G4LambertianRand(theGlobalNormal);
350           theFacetNormal = (NewMomentum - OldMomentum).unit();
351
352         }
353         else if ( theFinish == ground ) {
354
355           theStatus = LobeReflection;
356           if ( PropertyPointer1 && PropertyPointer2 ){
357           } else {
358              theFacetNormal =
359                  GetFacetNormal(OldMomentum,theGlobalNormal);
360           }
361           G4double PdotN = OldMomentum * theFacetNormal;
362           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
363
364         }
365         else {
366
367           theStatus = SpikeReflection;
368           theFacetNormal = theGlobalNormal;
369           G4double PdotN = OldMomentum * theFacetNormal;
370           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
371
372         }
373         G4double EdotN = OldPolarization * theFacetNormal;
374         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
375 }


epsilon:np blyth$ g4-hh G4LambertianRand
/usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/optical/include/G4OpBoundaryProcess.hh:          NewMomentum = G4LambertianRand(theGlobalNormal);
/usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/global/HEPRandom/include/G4RandomTools.hh:inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
epsilon:np blyth$

59 inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
 60 {
 61   G4ThreeVector vect;
 62   G4double ndotv;
 63   G4int count=0;
 64   const G4int max_trials = 1024;
 65
 66   do
 67   {
 68     ++count;
 69     vect = G4RandomDirection();
 70     ndotv = normal * vect;
 71
 72     if (ndotv < 0.0)
 73     {
 74       vect = -vect;
 75       ndotv = -ndotv;
 76     }
 77
 78   } while (!(G4UniformRand() < ndotv) && (count < max_trials));
 79
 80   return vect;
 81 }




