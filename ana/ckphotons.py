#!/usr/bin/env python

import math 
import numpy as np

from scipy.interpolate import interp1d 
from opticks.ana.mlib import GMaterialLib
mlib = GMaterialLib()

import matplotlib.pyplot as plt 


class G4Cerenkov(object):
    """
    391 void G4Cerenkov::BuildThePhysicsTable()
    392 {
    ...
    424             // Retrieve the first refraction index in vector
    425             // of (photon energy, refraction index) pairs 
    426 
    427             G4double currentRI = (*theRefractionIndexVector)[0];
    428 
    429             if (currentRI > 1.0) {
    430 
    431                // Create first (photon energy, Cerenkov Integral)
    432                // pair  
    433 
    434                G4double currentPM = theRefractionIndexVector->Energy(0);
    435                G4double currentCAI = 0.0;
    436 
    437                aPhysicsOrderedFreeVector->InsertValues(currentPM , currentCAI);
    438 
    439                // Set previous values to current ones prior to loop
    440 
    441                G4double prevPM  = currentPM;
    442                G4double prevCAI = currentCAI;
    443                G4double prevRI  = currentRI;
    444 
    445                // loop over all (photon energy, refraction index)
    446                // pairs stored for this material  
    447 
    448                for (size_t ii = 1;
    449                            ii < theRefractionIndexVector->GetVectorLength();
    450                            ++ii) {
    451                    currentRI = (*theRefractionIndexVector)[ii];
    452                    currentPM = theRefractionIndexVector->Energy(ii);
    453 
    454                    currentCAI = 0.5*(1.0/(prevRI*prevRI) +
    455                                      1.0/(currentRI*currentRI));
    456 
    457                    currentCAI = prevCAI + (currentPM - prevPM) * currentCAI;
    458 
    459                    aPhysicsOrderedFreeVector->
    460                                          InsertValues(currentPM, currentCAI);
    461 
    462                    prevPM  = currentPM;
    463                    prevCAI = currentCAI;
    464                    prevRI  = currentRI;
    465                }
    466 
    467             }
    ...
    """
    @classmethod
    def CerenkovAngleIntegral(cls, ev, ri):
        """
        :param ev: np.ndarray domain of energy (eV) in ascending order
        :param ri: np.ndarry values of RINDEX on the domain
        :return cai:  np.ndarray numerical 1/ri^2 energy integral starting at zero
        """
        assert len(ev) == len(ri)
        x = ev
        y = 1./(ri*ri)
        ymid = (y[:-1]+y[1:])/2.                     # looses entry as needs pair
        xdif = np.diff(x)                            # also looses entry 
        cai = np.zeros( len(ri), dtype=np.float32 )  # gains entry from first zero 
        np.cumsum(ymid*xdif, out=cai[1:] )
        return cai 
    pass



if __name__ == '__main__':

    ri0 = mlib("Water.RINDEX").copy()

    # energy(eV) and refractive index in ascending energy 
    ev = mlib.ev[::-1]  
    ri = ri0[::-1]

    fig, ax = plt.subplots(1)
    ax.plot( ev, ri, "o" )
    fig.show()


if 0:

    cai = G4Cerenkov.CerenkovAngleIntegral(ev, ri) 

    eV = 1.0e-06
    cm = 10.  
    Rfact = 369.81/(eV * cm)

    MeV = 1. 
    GeV = 1000.

    mass_c2 = {} 
    mass_c2["electron"] = 0.510998910 * MeV    
    mass_c2["proton"] = 938.272013 * MeV
    mass_c2["muon"] = 105.659 * MeV

    eplus = 1.0
    charge = eplus

    e_total = 200.*GeV
    e_rest = mass_c2["muon"] 
    gamma = e_total/e_rest
    beta = math.sqrt( 1. - 1./(gamma*gamma) )
    BetaInverse = 1./beta

    CAImax = cai.max()

    Pmin = ev[0]
    Pmax = ev[-1]

    nMin = ri.min()
    nMax = ri.max()

    print("%-20s : %s " % ("BetaInverse", BetaInverse))
    print("%-20s : %s " % ("nMin", nMin))
    print("%-20s : %s " % ("nMax", nMax))

    if nMax < BetaInverse: 
        dp = 0.0
        ge = 0.0
    elif nMin > BetaInverse:
        dp = Pmax - Pmin
        ge = CAImax
    else:
        """
        //  If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
        // we need to find a P such that the value of n(P) == 1/Beta.
        // Interpolation is performed by the GetEnergy() and
        // Value() methods of the G4MaterialPropertiesTable and
        // the GetValue() method of G4PhysicsVector.  
        //Pmin = Rindex->GetEnergy(BetaInverse);
        """
        assert np.all( np.diff(ri) > 0. )   # usually ri is not monotonic
        Pmin = np.interp( BetaInverse, ri, ev )
        dp = Pmax - Pmin

        """
        // need boolean for current implementation of G4PhysicsVector
        // ==> being phased out
        G4bool isOutRange;
        G4double CAImin = CerenkovAngleIntegrals->GetValue(Pmin, isOutRange);
        ge = CAImax - CAImin;
        """
        assert np.all( np.diff(ev) > 0. )
        CAImin = np.interp( Pmin, ev, cai )
        ge = CAImax - CAImin 
    pass
    NumPhotons = Rfact * charge/eplus * charge/eplus * (dp - ge * BetaInverse*BetaInverse)


 

    

