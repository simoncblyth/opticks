#include "G4PhysicsOrderedFreeVector.hh"
#include "G4MaterialPropertyVector.hh"
#include "X4ScintillationIntegral.hh"


G4PhysicsOrderedFreeVector* X4ScintillationIntegral::Integral( const G4MaterialPropertyVector* theFastLightVector )
{
     G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector = new G4PhysicsOrderedFreeVector();

          if (theFastLightVector) { 

               G4double currentIN = (*theFastLightVector)[0];

                if (currentIN >= 0.0) {

                    // Create first (photon energy, Scintillation 
                    // Integral pair  

                    G4double currentPM = theFastLightVector->
                        Energy(0);

                    G4double currentCII = 0.0;

                    aPhysicsOrderedFreeVector->
                        InsertValues(currentPM , currentCII);

                    // Set previous values to current ones prior to loop

                    G4double prevPM  = currentPM;
                    G4double prevCII = currentCII;
                    G4double prevIN  = currentIN;

                    // loop over all (photon energy, intensity)
                    // pairs stored for this material  

                    for(size_t ii = 1;
                              ii < theFastLightVector->GetVectorLength();
                              ++ii)
                    {
                        currentPM = theFastLightVector->Energy(ii);

                        currentIN= (*theFastLightVector)[ii];

                        currentCII = 0.5 * (prevIN + currentIN);

                        currentCII = prevCII +
                            (currentPM - prevPM) * currentCII;

                        aPhysicsOrderedFreeVector->
                            InsertValues(currentPM, currentCII);

                        prevPM  = currentPM;
                        prevCII = currentCII;
                        prevIN  = currentIN;
                    }
               }
            }

    return aPhysicsOrderedFreeVector ; 
}


