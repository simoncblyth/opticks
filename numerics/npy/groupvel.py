#!/usr/bin/env python
"""


* :google:`geant4 bug report 741`

* http://bugzilla-geant4.kek.jp/show_bug.cgi?id=741

Optical photons incorrectly propagating at phase velocity, should be group velocity



G. Horton-Smith, 2005/04/14
-----------------------------

* http://neutrino.phys.ksu.edu/~gahs/G4_GROUPVEL_fix/
* http://neutrino.phys.ksu.edu/~gahs/G4_GROUPVEL_fix/G4Track.patch

As described in Geant4 bug report #741, optical photons in Geant4 release 7.0
propagate at the phase velocity c/n(E), where E is the energy of the photon.
This is a bug because photons in real life propagate at the group velocity 
vg = c/(n(E)+dn/d(log(E)).


Geant4 fix into 4.7.1
-----------------------

* http://geant4.web.cern.ch/geant4/support/ReleaseNotes4.7.1.html

Added SetGROUPVEL() to G4MaterialPropertiesTable. Addresses problem report #741.



Observe
--------

* https://www.physicsforums.com/threads/trying-to-derive-a-group-velocity-equation.441274/


Is that the same ? ::

   d(log(E))     1
  ---------  =  --  
     dE          E

* https://en.wikipedia.org/wiki/Dispersion_(optics)

    vg = c / ( n - w dn/dw )



::

    034 // File: G4MaterialPropertiesTable.cc 
     35 // Version:     1.0
     36 // Created:     1996-02-08
     37 // Author:      Juliet Armstrong
     38 // Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
     39 //              Horton-Smith (bug report #741), by P. Gumplinger


    119 G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    120 {
    ...
    141   G4MaterialPropertyVector* groupvel = new G4MaterialPropertyVector();
    142 
    143   // fill GROUPVEL vector using RINDEX values
    144   // rindex built-in "iterator" was advanced to first entry above
    145   //
    146   G4double E0 = rindex->Energy(0);
    147   G4double n0 = (*rindex)[0];
    ...
    160     G4double E1 = rindex->Energy(1);
    161     G4double n1 = (*rindex)[1];
    168 
    169     G4double vg;
    170 
    171     // add entry at first photon energy
    172     //
    173     vg = c_light/(n0+(n1-n0)/std::log(E1/E0));
    174 
    175     // allow only for 'normal dispersion' -> dn/d(logE) > 0
    176     //
    177     if((vg<0) || (vg>c_light/n0))  { vg = c_light/n0; }
    178 
    179     groupvel->InsertValues( E0, vg );
    180 
    181     // add entries at midpoints between remaining photon energies
    182     //
    183 
    184     for (size_t i = 2; i < rindex->GetVectorLength(); i++)
    185     {
    186       vg = c_light/( 0.5*(n0+n1)+(n1-n0)/std::log(E1/E0));
    187 
    188       // allow only for 'normal dispersion' -> dn/d(logE) > 0
    189       //
    190       if((vg<0) || (vg>c_light/(0.5*(n0+n1))))  { vg = c_light/(0.5*(n0+n1)); }
    191       groupvel->InsertValues( 0.5*(E0+E1), vg );
    192 
    193       // get next energy/value pair, or exit loop
    194       //
    195       E0 = E1;
    196       n0 = n1;
    197       E1 = rindex->Energy(i);
    198       n1 = (*rindex)[i];
    199 
    200       if (E1 <= 0.)
    201       {
    202         G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    203                     FatalException, "Optical Photon Energy <= 0");
    204       }
    205     }
    206 
    207     // add entry at last photon energy
    208     //
    209     vg = c_light/(n1+(n1-n0)/std::log(E1/E0));
    210 
    211     // allow only for 'normal dispersion' -> dn/d(logE) > 0
    212     //
    213     if((vg<0) || (vg>c_light/n1))  { vg = c_light/n1; }
    214     groupvel->InsertValues( E1, vg );
    215   }
    216   else // only one entry in RINDEX -- weird!
    217   {
    218     groupvel->InsertValues( E0, c_light/n0 );
    219   }
    220 
    221   this->AddProperty( "GROUPVEL", groupvel );
    222 
    223   return groupvel;
    224 }


::

    In [22]: np.dstack([w, n])
    Out[22]: 
    array([[[  60.   ,    1.434],
            [  79.737,    1.434],
            [  99.474,    1.434],
            [ 119.211,    1.434],
            [ 138.947,    1.642],




Negative is normal dispersion (n down and w up)::

    In [26]: 1000.*dn/dw
    Out[26]: 
    array([  0.   ,   0.   ,   0.   ,  10.542,   5.896, -12.743,   4.491,
            -0.933,  -0.933,  -0.933,  -0.933,  -0.933,  -0.264,  -0.264,
            -0.264,  -0.264,  -0.264,  -0.144,  -0.105,  -0.095,  -0.095,
            -0.072,  -0.062,  -0.062,  -0.06 ,  -0.059,  -0.048,  -0.039,
            -0.039,  -0.039,  -0.039,  -0.028,  -0.016,  -0.016,  -0.016,
            -0.016,  -0.016,   0.   ])

    In [27]: n
    Out[27]: 
    array([ 1.434,  1.434,  1.434,  1.434,  1.642,  1.758,  1.507,  1.596,
            1.577,  1.559,  1.54 ,  1.522,  1.503,  1.498,  1.493,  1.488,
            1.483,  1.477,  1.475,  1.473,  1.471,  1.469,  1.467,  1.466,
            1.465,  1.464,  1.463,  1.462,  1.461,  1.46 ,  1.459,  1.459,
            1.458,  1.458,  1.457,  1.457,  1.457,  1.456,  1.456], dtype=float32)




* https://en.wikipedia.org/wiki/Talk%3ADispersion_(optics)

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/420/1.html

* https://indico.fnal.gov/contributionDisplay.py?sessionId=18&contribId=41&confId=4535


"""

from env.numerics.npy.PropLib import PropLib

if __name__ == '__main__':

    mlib = PropLib("GMaterialLib")

    mat = mlib("MineralOil")

    n = mat[:,0]
  
    w = mlib.domain

    dn = n[1:] - n[0:-1]

    dw = w[1:] - w[0:-1]



