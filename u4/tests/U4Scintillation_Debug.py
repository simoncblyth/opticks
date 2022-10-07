#!/usr/bin/env python
"""
U4Scintillation_Debug.py
==========================

DsG4Scintillation.cc::

     358         //////////////////////////////////// Birks' law ////////////////////////
     359         // J.B.Birks. The theory and practice of Scintillation Counting. 
     360         // Pergamon Press, 1964.      
     361         // For particles with energy much smaller than minimum ionization 
     362         // energy, the scintillation response is non-linear because of quenching  
     363         // effect. The light output is reduced by a parametric factor: 
     364         // 1/(1 + birk1*delta + birk2* delta^2). 
     365         // Delta is the energy loss per unit mass thickness. birk1 and birk2 
     366         // were measured for several organic scintillators.         
     367         // Here we use birk1 = 0.0125*g/cm2/MeV and ignore birk2.               
     368         // R.L.Craun and D.L.Smith. Nucl. Inst. and Meth., 80:239-244, 1970.   
     369         // Liang Zhan  01/27/2006 
     370         // /////////////////////////////////////////////////////////////////////
     371 

Light output reduced by parametric factor::

    1/(1 + birk1*delta + birk2* delta^2)


* birk1 : 0.0125*g/cm2/MeV 
* delta : 



G4double dE = TotalEnergyDeposit;
G4double dx = aStep.GetStepLength();
G4double dE_dx = dE/dx;             // energy over length 
G4double delta = dE_dx/aMaterial->GetDensity();//get scintillator density 






"""

import os, re, numpy as np

class _U4Scintillation_Debug(object):
    PKGDIR = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
    PTN = re.compile("^    double (\S*) ;$")
    @classmethod
    def Parse(cls):
        path = os.path.join(cls.PKGDIR, "U4Scintillation_Debug.hh")
        lines = open(path).read().splitlines()
        fields = []
        for line in lines:
            match = cls.PTN.match(line)
            if match:
                g = match.groups()[0]
                fields.append(g)
            pass
        pass
        return fields 
 

class U4Scintillation_Debug(object):
    BASE = "/tmp/u4debug"
    NAME = "U4Scintillation_Debug.npy"
    FIELDS = _U4Scintillation_Debug.Parse()

    def __init__(self, rel, symbol):
        path = os.path.join(self.BASE, rel, self.NAME)
        self.a = np.load(path)
        self.rel = rel
        self.symbol = symbol
        self.path = path 

    def __repr__(self):
        return "%s %s %s %s" % (self.symbol, self.rel, str(self.a.shape), self.path)  


    def QuenchedTotalEnergyDeposit(self, idx):
        d = self[idx]
        return d["TotalEnergyDeposit"]/(1.+d["birk1"]*d["delta"])  

    def QuenchedTotalEnergyDeposit_untypo(self, idx):
        d = self[idx]
        return d["TotalEnergyDeposit"]/(1.+d["birk1"]*1e-6*d["delta"])  

    def MeanNumberOfPhotons(self, idx):
        d = self[idx]
        return d["ScintillationYield"]*d["QuenchedTotalEnergyDeposit"]                                                                                   

    def MeanNumberOfPhotons_untypo(self, idx):
        d = self[idx]
        return d["ScintillationYield"]*self.QuenchedTotalEnergyDeposit_untypo(idx)

    def __getitem__(self, idx):
        vals = self.a[idx].ravel()
        keys = self.FIELDS
        return dict(zip(keys, vals))



if __name__ == '__main__':
    s30 = U4Scintillation_Debug("ntds3/000", "s30" )
    s31 = U4Scintillation_Debug("ntds3/001", "s31" )

    print(s30)
    print(s31)
    print(U4Scintillation_Debug.FIELDS)




