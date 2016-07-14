ana-rel(){      echo ana ; }
ana-src(){      echo ana/ana.bash ; }
ana-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ana-src)} ; }
ana-vi(){       vi $(ana-source) ; }
ana-usage(){ cat << \EOU

Opticks Analysis Scripts
=========================

PMT Tests
------------

:doc:`pmt_test_evt`
    Loads single PMT test event.

:doc:`pmt_test`
    Compare Opticks and Geant4 photon bounce histories for simple PMT in box of mineral oil geometry 

:doc:`pmt_test_distrib`
    Compare Opticks and Geant4 photon distributions for simple PMT in box of mineral oil geometry 

:doc:`pmt_skimmer`
    Plotting mean step by step positions of all photons with specific step histories 

**pmt_edge.py**
    tabulates and plots and min,max,mid positions at each step for photons that 
    follow a specific sequence such as "TO BT BT SA" 


Rainbow Tests
---------------

:doc:`source`
    Compare wavelength spectrum from ggv-rainbow against analytic Planck distribution

**planck.py**
    Planck black body formula

**droplet.py**
    geometry calculation of spherical drop incident, refracted, 
    deviation angles for k orders of rainbow corresponding to different numbers
    of internal reflections

**sphere.py**
    SphereReflect intersection, polarization calculation and spatial plot

G4Gun Tests
--------------

:doc:`g4gun`
    Load single G4Gun event


Geometry Infrastructure 
-------------------------

:doc:`boundary`
    Access to geocache material properties and boundary holder class

:doc:`proplib`
    Access to geocache via PropLib
 
:doc:`mergedmesh`
    Access geometrical data such as positions, transforms of volumes of the geometry

**geometry.py**
    Shape, Ray, Plane, Intersect, IntersectFrame : simple intersect calulations


Event Infrastructure
-----------------------

**evt.py**
    loads event data

**base.py** 
    internal envvar setup based on input envvar IDPATH 
    json and ini loading with Abbrev and ItemList classes 

**nload.py**
    numpy array and .ini metadata loading with clases A, I, II

**ncensus.py**
    event census with array shape dumping 

**nbase.py**
    pure numpy utility functions: count_unique, count_unique_sorted, chi2, decompression_bins 

**seq.py**
    SeqType conversions of long integer sequence codes to abbreviation string sequences like  "TO BT BR BR BT SA"
    SeqTable presenting frequencies of different sequences 

**history.py**
    HisType (SeqType subclass) and tests
        
**material.py**
    MatType (SeqType subclass) and tests

**metadata.py** 
    access metadata .json written by Opticks allowing 
    comparisons of evt digests and simulation times 

**genstep.py** 
    fit genstep xyz vs time, to obtain parametric eqn for the viewpoint tracking 
    used to create videos, see `vids-`


Color Infrastructure
----------------------

**cie.py**
    converts wavelength spectra into XYZ and RGB colorspace (depends on **env** repo)




EOU
}

ana-notes(){ cat << EON

box_test.py

cfg4_speedplot.py

cfplot.py



REFLECTION CHECKS

fresnel.py
      analytic reflection expectations from Fresnel formula
reflection.py
      comparison of simulated S and P absolute reflection with fresnel formula 
      produced by::

           ggv-reflect --spol
           ggv-reflect --ppol

      NEEDS DEBUG : NOT MATCHING ANALYTIC ANYMORE   
        

PRISM CHECKS

prism.py
prism_spectrum.py


RAINBOW CHECKS

xrainbow.py
     XRainbow expectations 
rainbow.py
rainbowplot.py

rainbow_cfg4.py
rainbow_check.py
rainbow_scatter.py



DEBUGGING


types.py
     MOSTLY DEPRECATED FUNCTIONS

analytic_cf_triangulated.py
     plotting analytic PMT and mesh points together

nopstep_viz_debug.py
     creates fake nopstep (non-photon step) for visualization debugging

mesh.py
     debugging mesh structure, comparing multiple meshes

dae.py
     simple XML COLLADA parsing of .dae for debugging 

xmatlib.py
     COLLADA XML material properties access for debugging

groupvel.py
     debug in progress

genstep_sequence_material_mismatch.py
     dbg 

CGDMLDetector.py
     access to global transforms obtained from GDMLDetector  

instancedmergedmesh.py
     debugging of instanced mesh info such as PMT meshes  

vacuum_offset.py
     geometry debug, the sagging vacuum

IndexerTest.py
     debug small GPU CPU indexing differences 

polarization.py
     checking magnitude of polarization for Opticks and G4 rainbow events


EON
}




ana-env(){
    olocal-
    opticks-
}

ana-sdir(){ echo $(opticks-home)/ana ; }
ana-tdir(){ echo $(opticks-home)/ana/tests ; }
ana-idir(){ echo $(opticks-idir); }
ana-bdir(){ echo $(opticks-bdir)/$(ana-rel) ; }

ana-cd(){   cd $(ana-sdir); }
ana-scd(){  cd $(ana-sdir); }
ana-tcd(){  cd $(ana-tdir); }
ana-icd(){  cd $(ana-idir); }
ana-bcd(){  cd $(ana-bdir); }

ana-name(){ echo Ana ; }
ana-tag(){  echo ANA ; }





