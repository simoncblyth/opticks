ana-rel(){      echo ana ; }
ana-src(){      echo ana/ana.bash ; }
ana-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ana-src)} ; }
ana-vi(){       vi $(ana-source) ; }
ana-usage(){ cat << \EOU

Opticks Analysis Scripts
=========================

:doc:`proplib`
     Access to geocache via PropLib, Material and Boundary classes
 
:doc:`mergedmesh`
     Access geometrical data such as positions, transforms of volumes of the geometry.

EOU
}

ana-notes(){ cat << EON

evt.py
    loads event data

base.py 
    internal envvar setup based on input envvar IDPATH 
    json and ini loading with Abbrev and ItemList classes 

nload.py
    numpy array and .ini metadata loading with clases A, I, II

nbase.py
    pure numpy utility functions: count_unique, count_unique_sorted, chi2, decompression_bins 


droplet.py
     geometry calculation of spherical drop incident, refracted, 
     deviation angles for k orders of rainbow corresponding to different numbers
     of internal reflections


seq.py
    SeqType conversions of long integer sequence codes to abbreviation string sequences like  "TO BT BR BR BT SA"
    SeqTable presenting frequencies of different sequences 

history.py
     HisType (SeqType subclass) and tests
        
material.py
     MatType (SeqType subclass) and tests




metadata.py 
     access metadata .json written by Opticks allowing 
     comparisons of evt digests and simulation times 

mesh.py
     env dependency 


analytic_cf_triangulated.py
box_test.py
cfg4_speedplot.py
cfplot.py
cie.py
g4gun.py
genstep.py
geometry.py

ncensus.py
nopstep_viz_debug.py


REFLECTION CHECKS

fresnel.py
      analytic reflection expectations from Fresnel formula
reflection.py
      comparison of simulated S and P absolute reflection with fresnel formula 
      produced by::

           ggv-reflect --spol
           ggv-reflect --ppol

      NEEDS DEBUG : NOT MATCHING ANALYTIC ANYMORE   
        


PMT CHECKS

pmt_edge.py
pmt_skimmer.py
pmt_test.py
pmt_test_distrib.py
pmt_test_evt.py


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


source.py
sphere.py
torchevt.py
types.py


DEBUGGING

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





