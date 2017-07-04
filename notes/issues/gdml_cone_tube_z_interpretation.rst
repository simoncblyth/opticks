GDML cone/tube z-interpretation
==================================


gdml-pdf::

    The GDML Cone Segment is formed using 7 dimensions.

    rmin1 
         inner radius at base of cone 
    rmax1 
         outer radius at base of cone 
    rmin2 
         inner radius at top of cone 
    rmax2 
         outer radius at top of cone
    z 
         height of cone segment   : ? is that -z/2->z/2  or 0->z  
    startphi 
         start angle of the segment 
    deltaphi 
         angle of the segment


    The GDML Tube Segment is formed using 5 dimensions:

    rmin 
         inside radius of segment - if not given 0.0 is defaulted
    rmax 
         outside radius of segment
    z 
         z length of tube segment  : ? -z/2->z/2 or 0->z 
    startphi 
         starting phi position angle of segment - if not given 0.0 is defaulted deltaphi delta angle of segment
   

g4-cls G4GDMLReadSolids::

    1762 void G4GDMLReadSolids::TubeRead(const xercesc::DOMElement* const tubeElement)
    1763 {
    1764    G4String name;
    1765    G4double lunit = 1.0;
    1766    G4double aunit = 1.0;
    1767    G4double rmin = 0.0;
    1768    G4double rmax = 0.0;
    1769    G4double z = 0.0;
    1770    G4double startphi = 0.0;
    1771    G4double deltaphi = 0.0;
    ....
    1793       const G4String attName = Transcode(attribute->getName());
    1794       const G4String attValue = Transcode(attribute->getValue());
    1795 
    1796       if (attName=="name") { name = GenerateName(attValue); } else
    1797       if (attName=="lunit") { lunit = G4UnitDefinition::GetValueOf(attValue); } else
    1798       if (attName=="aunit") { aunit = G4UnitDefinition::GetValueOf(attValue); } else
    1799       if (attName=="rmin") { rmin = eval.Evaluate(attValue); } else
    1800       if (attName=="rmax") { rmax = eval.Evaluate(attValue); } else
    1801       if (attName=="z") { z = eval.Evaluate(attValue); } else
    1802       if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
    1803       if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); }
    1804    }
    1805 
    1806    rmin *= lunit;
    1807    rmax *= lunit;
    1808    z *= 0.5*lunit;
    1809    startphi *= aunit;
    1810    deltaphi *= aunit;
    1811 
    1812    new G4Tubs(name,rmin,rmax,z,startphi,deltaphi);
    1813 }


g4-cls G4Tubs::

    034 // G4Tubs
     35 //
     36 // Class description:
     37 //
     38 //   A tube or tube segment with curved sides parallel to
     39 //   the z-axis. The tube has a specified half-length along
     40 //   the z-axis, about which it is centered, and a given
     41 //   minimum and maximum radius. A minimum radius of 0
     42 //   corresponds to filled tube /cylinder. The tube segment is
     43 //   specified by starting and delta angles for phi, with 0
     44 //   being the +x axis, PI/2 the +y axis.
     45 //   A delta angle of 2PI signifies a complete, unsegmented
     46 //   tube/cylinder.
     47 //
     48 //   Member Data:
     49 //
     50 //   fRMin  Inner radius
     51 //   fRMax  Outer radius
     52 //   fDz  half length in z
     53 //


    0223 void G4GDMLReadSolids::ConeRead(const xercesc::DOMElement* const coneElement)
     224 {
     225    G4String name;
     226    G4double lunit = 1.0;
     227    G4double aunit = 1.0;
     228    G4double rmin1 = 0.0;
     229    G4double rmax1 = 0.0;
     230    G4double rmin2 = 0.0;
     231    G4double rmax2 = 0.0;
     232    G4double z = 0.0;
     233    G4double startphi = 0.0;
     234    G4double deltaphi = 0.0;
     ...
     259       if (attName=="name") { name = GenerateName(attValue); } else
     260       if (attName=="lunit") { lunit = G4UnitDefinition::GetValueOf(attValue); } else
     261       if (attName=="aunit") { aunit = G4UnitDefinition::GetValueOf(attValue); } else
     262       if (attName=="rmin1") { rmin1 = eval.Evaluate(attValue); } else
     263       if (attName=="rmax1") { rmax1 = eval.Evaluate(attValue); } else
     264       if (attName=="rmin2") { rmin2 = eval.Evaluate(attValue); } else
     265       if (attName=="rmax2") { rmax2 = eval.Evaluate(attValue); } else
     266       if (attName=="z") { z = eval.Evaluate(attValue); } else
     267       if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
     268       if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); }
     269    }
     270 
     271    rmin1 *= lunit;
     272    rmax1 *= lunit;
     273    rmin2 *= lunit;
     274    rmax2 *= lunit;
     275    z *= 0.5*lunit;
     276    startphi *= aunit;
     277    deltaphi *= aunit;
     278 
     279    new G4Cons(name,rmin1,rmax1,rmin2,rmax2,z,startphi,deltaphi);
     280 }

g4-cls G4Cons::

     35 // Class description:
     36 //
     37 //   A G4Cons is, in the general case, a Phi segment of a cone, with
     38 //   half-length fDz, inner and outer radii specified at -fDz and +fDz.
     39 //   The Phi segment is described by a starting fSPhi angle, and the
     40 //   +fDPhi delta angle for the shape.
     41 //   If the delta angle is >=2*pi, the shape is treated as continuous
     42 //   in Phi
     43 //
     44 //   Member Data:
     45 //
     46 //  fRmin1  inside radius at  -fDz
     47 //  fRmin2  inside radius at  +fDz
     48 //  fRmax1  outside radius at -fDz
     49 //  fRmax2  outside radius at +fDz
     50 //  fDz  half length in z
     



