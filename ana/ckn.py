#!/usr/bin/env python
"""
ckn.py : reproduce G4Cerenkov_modified::GetAverageNumberOfPhotons 
====================================================================

::

    ipython -i ckn.py 


"""
import os, logging, numpy as np
import matplotlib.pyplot as plt
from opticks.ana.main import opticks_main
from opticks.ana.key import keydir
from opticks.ana.nload import np_load

from scipy import integrate 


log = logging.getLogger(__name__)

class CKN(object):
    """
    Reproduces the G4Cerenkov Frank-Tamm integration to give average number of photons
    for a BetaInverse and RINDEX profile.
    """
    kd = keydir()
    rindex_path = os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy")

    def __init__(self):
        ri = np.load(self.rindex_path)
        ri[:,0] *= 1e6    # into eV 
        ri_ = lambda ev:np.interp( ev, ri[:,0], ri[:,1] ) 

        self.ri = ri
        self.ri_ = ri_
        self.BuildThePhysicsTable()
        self.BuildThePhysicsTable_2()
        assert np.allclose( self.cai, self.cai2 )

    def BuildThePhysicsTable(self, dump=False):
        """
        See G4Cerenkov_modified::BuildThePhysicsTable


        This is applying the composite trapezoidal rule to do a 
        numerical energy integral  of  n^(-2) = 1./(ri[:,1]*ri[:,1])
        """
        ri = self.ri
        en = ri[:,0]

        ir2 = 1./(ri[:,1]*ri[:,1])
        mir2 = 0.5*(ir2[1:] + ir2[:-1])  

        de = en[1:] - en[:-1]

        assert len(mir2) == len(ri) - 1       # averaging points looses one value 
        mir2_de = mir2*de 

        cai = np.zeros(len(ri))               # leading zero regains one value 
        np.cumsum(mir2_de, out=cai[1:])  

        if dump:
            print("cai", cai)
        pass

        self.cai = cai
        self.ir2 = ir2
        self.mir2 = mir2
        self.de = de
        self.mir2_de = mir2_de


    def BuildThePhysicsTable_2(self, dump=False):
        """
        np.trapz does the same thing as above : applying composite trapezoidal integration

        https://numpy.org/doc/stable/reference/generated/numpy.trapz.html
        """
        ri = self.ri
        en = ri[:,0]
        ir2 = 1./(ri[:,1]*ri[:,1])

        cai2 = np.zeros(len(ri))
        for i in range(len(ri)):
            cai2[i] = np.trapz( ir2[:i+1], en[:i+1] ) 
        pass
        self.cai2 = cai2

        if dump:
            print("cai2", cai2)
        pass


    @classmethod
    def BuildThePhysicsTable_s2i(cls, ri, BetaInverse, dump=False):
        """
        """
        en = ri[:,0]
        s2i = np.zeros(len(ri))
        for i in range(len(ri)):
            s2i[i] = np.trapz( s2[:i+1], en[:i+1] ) 
        pass
        return s2i

    def GetAverageNumberOfPhotons_s2(self, BetaInverse, charge=1, dump=False ):
        """
        Simplfied Alternative to _s2messy following C++ implementation. 
        Allowed regions are identified by s2 being positive avoiding the need for 
        separately getting crossings. Instead get the crossings and do the trapezoidal 
        numerical integration in one pass, improving simplicity and accuracy.  
    
        See opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc
        """
        s2integral = 0.
        for i in range(len(self.ri)-1):
            en = np.array([self.ri[i,0], self.ri[i+1,0] ]) 
            ri = np.array([self.ri[i,1], self.ri[i+1,1] ]) 
            ct = BetaInverse/ri
            s2 = (1.-ct)*(1.+ct) 

            if s2[0] <= 0. and s2[1] <= 0.:
                pass
            elif s2[0] < 0. and s2[1] > 0.:
                en_cross = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0])
                s2_cross = 0.
                s2integral +=  (en[1] - en_cross)*(s2_cross + s2[1])*0.5
            elif s2[0] >= 0. and s2[1] >= 0.:
                s2integral += (en[1] - en[0])*(s2[0] + s2[1])*0.5
            elif s2[0] > 0. and s2[1] < 0.:
                en_cross = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0]) 
                s2_cross = 0. 
                s2integral +=  (en_cross - en[0])*(s2_cross + s2[0])*0.5
            else:
                print( " en_0 %10.5f ri_0 %10.5f s2_0 %10.5f  en_1 %10.5f ri_1 %10.5f s2_1 %10.5f " % (en[0], ri[0], s2[0], en[1], ri[1], s2[1] )) 
                assert 0 
            pass
        pass
        Rfact = 369.81 / 10. #        Geant4 mm=1 cm=10    
        NumPhotons = Rfact * charge * charge * s2integral
        return NumPhotons 

    def GetAverageNumberOfPhotons_s2messy(self, BetaInverse, charge=1, dump=False ):
        """
        NB see GetAverageNumberOfPhotons_s2 it gives exactly the same results as this and is simpler

        Alternate approach doing the numerical integration directly of s2 rather than 
        doing it on n^-2 and combining it later with the integral of the 1 and the 
        BetaInverse*BetaInverse

        Doing the integral of s2 avoids inconsistencies in the numerical approximations
        which prevents the average number of photons going negative in the region when the 
        BetaInverse "sea level" rises to almost engulf the last rindex peak.:: 

                                                  BetaInverse*BetaInverse
              Integral ( s2 )  = Integral(  1 - ---------------------------  )  = Integral ( 1 - c2 )
                                                          ri*ri 
        """
        self.BetaInverse = BetaInverse
        ckn = self
        ri = ckn.ri
        en = ri[:,0]

        s2 = np.zeros( (len(ri), 2), dtype=np.float64 )
        ct = BetaInverse/ri[:,1]
        s2[:,0] = ri[:,0]
        s2[:,1] = (1. - ct)*(1. + ct )

        cross = ckn.FindCrossings( s2, 0. )
        s2integral = 0. 
        for i in range(len(cross)//2):
            en0 = cross[2*i+0]
            en1 = cross[2*i+1]
            # select bins within the range 
            s2_sel = s2[np.logical_and(s2[:,0]>=en0, s2[:,0] <= en1)] 

            # fabricate partial bins before and after the full ones 
            # that correspond to s2 zeros 
            fs2 = np.zeros( (2+len(s2_sel),2), dtype=np.float64 )  
            fs2[0] = [en0, 0.]
            fs2[1:-1] = s2_sel
            fs2[-1] = [en1, 0.]

            s2integral += np.trapz( fs2[:,1], fs2[:,0] )   # trapezoidal integration
        pass
        Rfact =  369.81   #  (eV * cm)^-1
        Rfact *= 0.1      # cm to mm ?  Geant4: mm = 1. cm = 10.  

        NumPhotons = Rfact * charge * charge * s2integral
        self.NumPhotons = NumPhotons
        if dump:
            print(" s2integral %10.4f " % (s2integral)) 
        pass
        return NumPhotons

    def GetAverageNumberOfPhotons_asis(self, BetaInverse, charge=1, dump=False ):
        """
        This duplicates the results from G4Cerenkov_modified::GetAverageNumberOfPhotons
        including negative numbers of photons for BetaInverse close to the rindex peak.

        Frank–Tamm formula gives number of Cerenkov photons per mm as an energy integral::

                                                        BetaInverse^2    
              N_photon  =     370.     Integral ( 1 - -----------------  )  dE      
                                                          ri(E)^2      

        Where the integration is over regions where :    ri(E) > BetaInverse 
        which corresponds to a real cone angle and the above bracket being positive::
        
                        BetaInverse
              cos th = --------------   < 1  
                           ri(E) 

        The bracket above is in fact :    1 - cos^2 th = sin^2 th  which must be +ve 
        so getting -ve numbers of photons is clearly a bug from the numerical approximations
        being made.  Presumably the problem is due to the splitting of the integral into  
        CerenkovAngleIntegral "cai" which is the cumulative integral of   1./ri(E)^2
        followed by linear interpolation of this in order to get the integral between 
        crossings.

        G4Cerenkov::

             Rfact = 369.81/(eV * cm);

        https://www.nevis.columbia.edu/~haas/frank_epe_course/cherenkov.ps
        has 370(eV.cm)^-1

        ~/opticks_refs/nevis_cherenkov.ps

        hc = 1240 eV nm = 1240 eV cm * 1e-7    ( nm:1e-9 cm 1e-2)

        In [8]: 2*np.pi*1e7/(137*1240)     # fine-structure-constant 1/137 and hc = 1240 eV nm 
        Out[8]: 369.860213514221

        alpha/hc = 370 (eV.cm)^-1

        See ~/opticks/examples/UseGeant4/UseGeant4.cc UseGeant4::physical_constants::

            UseGeant4::physical_constants
                                                   eV 1e-06
                                                   cm 10
                                 fine_structure_const 0.00729735
                        one_over_fine_structure_const 137.036
              fine_structure_const_over_hbarc*(eV*cm) 369.81021
                      fine_structure_const_over_hbarc 36981020.84589
                            Rfact =  369.81/(eV * cm) 36981000.00000[as used by G4Cerenkov::GetAverageNumberOfPhotons] 
                                  2*pi*1e7/(1240*137) 369.86021
                                                eplus 1.00000
                                     electron_mass_c2 0.51099891
                                       proton_mass_c2 938.27201300
                                      neutron_mass_c2 939.56536000


        Crossing points from similar triangles:: 


             x - prevPM                            currentPM - prevPM
             ------------------------------   =     ------------------------ 
             BetaInverse - prevRI                   currentRI - prevRI 


             x - prevPM =  (BetaInverse-prevRI)/(currentRI-prevRI)*(currentPM-prevPM) 



                             (currentPM, currentRI)
                                +                   
                               /
                              /
                             /
                            /
                           /
                          /
                         *
                        /  (x,BetaInverse)
                       /                
                      /
                     /
                    /
                   +                                
            (prevPM, prevRI)            


        """
        self.BetaInverse = BetaInverse
        ri = self.ri
        cai = self.cai
        en = ri[:,0]

        cross = self.FindCrossings( ri, BetaInverse )
        if dump:
            print(" cross %s " % repr(cross) )
        pass
        self.cross = cross

        dp1 = 0.
        ge1 = 0. 

        for i in range(len(cross)//2):
            en0 = cross[2*i+0]
            en1 = cross[2*i+1]
            dp1 += en1 - en0

            # interpolating the cai is an approximation that is the probable cause of NumPhotons 
            # going negative for BetaInverse close to the "peak" of rindex

            cai0 = np.interp( en0, en, cai )
            cai1 = np.interp( en1, en, cai )
            ge1 += cai1 - cai0 
        pass
        Rfact =  369.81   #  (eV * cm)^-1
        Rfact *= 0.1      # cm to mm ?

        NumPhotons = Rfact * charge * charge * (dp1 - ge1 * BetaInverse*BetaInverse) 
 
        self.dp1 = dp1
        self.ge1 = ge1
        self.NumPhotons = NumPhotons

        if dump:
            print(" dp1 %10.4f ge1 %10.4f " % (dp1, ge1 )) 
        pass

        return NumPhotons

    @classmethod
    def FindCrossings(cls, pq, pv): 
        """
        :param pq: property array of shape (n,2)
        :param pv: scalar value
        :return cross: array of values where pv crosses the linear interpolated pq 
        """
        assert len(pq.shape) == 2 and pq.shape[1] == 2 

        mx = pq[:,1].max()
        mi = pq[:,1].min()

        cross = []
        if pv <= mi:
            cross.append( pq[0,0] )
            cross.append( pq[-1,0] )
        elif pv >= mx:
            pass
        else:
            if pq[0,1] >= pv:
                cross.append(pq[0,0])
            pass
            assert len(pq) > 2
            for ii in range(1,len(pq)-1):
                prevPM,prevRI = pq[ii-1]    
                currPM,currRI = pq[ii]
              
                down = prevRI >= pv and currRI < pv 
                up = prevRI < pv and currRI >= pv
                if down or up:
                    cross.append((pv-prevRI)/(currRI-prevRI)*(currPM-prevPM) + prevPM)
                pass
            pass
            if pq[-1,1] >= pv:
                cross.append(pq[-1,1])
            pass
        pass
        assert len(cross) % 2 == 0, cross
        return cross

    def test_GetAverageNumberOfPhotons(self, BetaInverse):
        NumPhotons_asis = self.GetAverageNumberOfPhotons_asis(BetaInverse)
        NumPhotons_s2 = self.GetAverageNumberOfPhotons_s2(BetaInverse)
        NumPhotons_s2messy = self.GetAverageNumberOfPhotons_s2messy(BetaInverse)
        fmt = "BetaInverse %6.4f _asis %6.4f  _s2 %6.4f _s2messy %6.4f    " 
        print( fmt % ( BetaInverse, NumPhotons_asis, NumPhotons_s2, NumPhotons_s2messy ))

    def scan_GetAverageNumberOfPhotons(self, x0=1., x1=2., nx=101 ):
        scan = np.zeros( (nx, 4), dtype=np.float64 )
        for i, BetaInverse in enumerate(np.linspace(x0, x1, nx )):
            NumPhotons_asis = self.GetAverageNumberOfPhotons_asis(BetaInverse)
            NumPhotons_s2 = self.GetAverageNumberOfPhotons_s2(BetaInverse)
            NumPhotons_s2messy = self.GetAverageNumberOfPhotons_s2messy(BetaInverse)
            scan[i] = [BetaInverse, NumPhotons_asis, NumPhotons_s2, NumPhotons_s2messy ]
            fmt = "  bi %7.3f _asis %7.3f _s2 %7.3f _s2messy %7.3f "  
            print( fmt % tuple(scan[i]) )
        pass
        self.scan = scan   

        path="/tmp/G4Cerenkov_modifiedTest/scan_GetAverageNumberOfPhotons.npy"
        if os.path.exists(path):
            self.scan2 = np.load(path)
        pass 



if __name__ == '__main__':

    ok = opticks_main()
    ckn = CKN()

    #ckn.test_GetAverageNumberOfPhotons(1.78)
    ckn.scan_GetAverageNumberOfPhotons()

    numPhoton_ = lambda bi:np.interp( bi, ckn.scan[:,0], ckn.scan[:,1] )    
    numPhotonS2_ = lambda bi:np.interp( bi, ckn.scan[:,0], ckn.scan[:,2] )    


    nMin = ckn.ri[:,1].min()  
    nMax = ckn.ri[:,1].max()  

    bi = [nMin, nMax]
    numPhotonMax = numPhotonS2_( np.linspace(bi[0], bi[1], 101) ).max()  # max in the BetaInverse range 


    fig, ax = plt.subplots(figsize=ok.figsize) 
    ax.set_xlim( *bi )
    ax.set_ylim(  -1., numPhotonMax )

    ax.scatter( ckn.scan[:,0], ckn.scan[:,1], label="GetAverageNumberOfPhotons", s=3 )
    ax.plot( ckn.scan[:,0], ckn.scan[:,1], label="GetAverageNumberOfPhotons" )

    ax.plot( ckn.scan[:,0], ckn.scan[:,2], label="GetAverageNumberOfPhotons_s2" )
    ax.scatter( ckn.scan[:,0], ckn.scan[:,2], label="GetAverageNumberOfPhotons_s2", s=3 )

    xlim = ax.get_xlim()
    ax.plot( xlim, [0,0], linestyle="dotted", label="zero" )


    ax.legend()
    fig.show()      





    fig, ax = plt.subplots(figsize=ok.figsize) 
    ax.set_xlim( *bi )
    #ax.set_ylim(  -1., numPhotonMax )

    ax.scatter( ckn.scan[:,0], ckn.scan[:,2] - ckn.scan[:,1], label="GetAverageNumberOfPhotons_s2 - GetAverageNumberOfPhotons", s=3 )
    ax.plot( ckn.scan[:,0], ckn.scan[:,2] - ckn.scan[:,1], label="GetAverageNumberOfPhotons_s2 - GetAverageNumberOfPhotons" )

    ylim = ax.get_ylim()

    for n in ckn.ri[:,1]:
        ax.plot( [n, n], ylim, label="%s" % n  ) 
    pass

    #ax.legend()
    fig.show()      









if 0:
    BetaInverse = ckn.BetaInverse

    cross = ckn.cross
    ri = ckn.ri
    cai = ckn.cai
    

    fig, axs = plt.subplots(1, 2, figsize=ok.figsize) 
 
    ax = axs[0]
    ax.plot(ri[:,0], ri[:,1] , label="linear interpolation" )
    ax.scatter(ri[:,0], ri[:,1], label="ri" )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot( xlim, [BetaInverse, BetaInverse], label="BetaInverse:%6.4f" % BetaInverse)
    for e in cross:
        ax.plot( [e, e], ylim, label="cross" )
    pass

    ax.legend() 

    ax = axs[1]
    ax.plot( ri[:,0], cai, label="cai" ) 
    ylim = ax.get_ylim()
    for e in cross:
        ax.plot( [e, e], ylim, label="cross" )
    pass
    ax.legend() 

    fig.show()



