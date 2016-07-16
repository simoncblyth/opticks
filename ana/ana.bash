ana-rel(){      echo ana ; }
ana-src(){      echo ana/ana.bash ; }
ana-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ana-src)} ; }
ana-vi(){       vi $(ana-source) ; }
ana-usage(){ cat << \EOU

ana : Opticks Analysis Scripts
=================================

General Tests
---------------

:doc:`tevt`
    Loads single event and dumps constituent array dimensions and photon history tables

PMT Tests
------------

:doc:`tpmt`
    Compare Opticks and Geant4 photon bounce histories for simple PMT in box of mineral oil geometry 

:doc:`tpmt_distrib`
    Compare Opticks and Geant4 photon distributions for simple PMT in box of mineral oil geometry 

:doc:`tpmt_skimmer`
    Plotting and tabulating mean step by step positions of all photons with specific step histories 
    such as "TO BT BT SA" 


BoxInBox tests
------------------

:doc:`tbox`
    BoxInBox Opticks vs Geant4 history sequence comparisons analogous to *pmt_test.py*

Rainbow Tests
---------------

:doc:`trainbow`
    Rainbow scattering angle comparison between Opticks and Geant4 

**xrainbow.py**
    Rainbow expectations with classes XRainbow and XFrac

**droplet.py**
    geometry calculation of spherical drop incident, refracted, 
    deviation angles for k orders of rainbow corresponding to different numbers
    of internal reflections

**sphere.py**
    SphereReflect intersection, polarization calculation and spatial plot


Source Tests
--------------

:doc:`source`
    Compare wavelength spectrum from ggv-rainbow against analytic Planck distribution

**planck.py**
    Planck black body formula


Prism Tests
-------------

**prism.py**
    Comparison of simulation with analytic expectation 

**prism_spectrum.py**
    Compare ggv-newton evts with PrismExpected


Reflection Tests
-------------------

**fresnel.py**
    analytic reflection expectations from Fresnel formula


**reflection.py**
    comparison of simulated S and P absolute reflection with Fresnel formula 
    produced by::

         ggv-reflect --spol
         ggv-reflect --ppol


G4Gun Tests
--------------

:doc:`g4gun`
    Load single G4Gun event


Geometry Infrastructure 
-------------------------

:doc:`material`
    Material class gives access to geocache material properties 

:doc:`boundary`
    Boundary class acts as holder of inner/outer materials and surfaces

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

**nbase.py**
    pure numpy utility functions: count_unique, count_unique_sorted, chi2, decompression_bins 

**seq.py**
    SeqType conversions of long integer sequence codes to abbreviation string sequences like  "TO BT BR BR BT SA"
    SeqTable presenting frequencies of different sequences 

**histype.py**
    HisType (SeqType subclass) and tests
        
**mattype.py**
    MatType (SeqType subclass) and tests

**genstep.py** 
    fit genstep xyz vs time, to obtain parametric eqn for the viewpoint tracking 
    used to create videos, see `vids-`

**ana.py**
    geometrical and plotting utils


Plotting Infrastructure
--------------------------

**cfplot.py**
    Comparison Plotter with Chi2 Underplot 


Metadata Infrastructure
--------------------------

**cfg4_speedplot.py**
    compare simulation times using json metadata written by Opticks simulation invokations

**metadata.py** 
    access metadata .json written by Opticks allowing 
    comparisons of evt digests and simulation times 

**ncensus.py**
    event census with array shape dumping 


Color Infrastructure
----------------------

**cie.py**
    converts wavelength spectra into XYZ and RGB colorspace (depends on **env** repo)



EOU
}

ana-notes(){ cat << EON



DEBUGGING : NOT SURFACING THESE IN THE DOCS


types.py
     MOSTLY DEPRECATED : INSTEAD USE histype mattype ETC

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


ana-py-(){
   ana-cd
   ls -1 *.py 
}

ana-skips-note(){ cat << EON
*ana-skips* 
    list .py scripts without executable bit set.

    Known fails, for example due to *env* repo dependency  
    can be skipped from ana-test simply by removing
    the executable bit::

        chmod ugo-x cie.py  

EON
}
ana-skips(){
   local py
   ana-py- | while read py ; do
      [ ! -x $py ] && echo $msg SKIP $py 
   done
}

ana-test-note(){ cat << EON
*ana-test*
     runs all executable .py in ana directory, 
     stopping at any script that yields a non-zero return code 

     Use to check for simple python breakages from moving 
     things around for example. 

     Intended to just test mechanics, not meaning.
     Failure to load events should not cause failures, just warnings.
EON
}
ana-test(){
   local py
   local rc
   ana-py- | while read py ; do
       [ ! -x $py ] && echo $msg SKIP $py && continue
       echo $msg $py

       ./$py
       rc=$?

       if [ "$rc" != "0" ]; then 
           echo $msg WARNING : PY FAILURE OF $py : RC $rc
           break 
       fi
   done   
}


ana-shim(){
  ana-cd 
  local py
  local rst
  local name
  ls -1  t*.py | while read py ; do
      rst=${py/.py}.rst
      if [ ! -f "$rst" ];  then
          echo $msg writing rst shim $rst shim 
          ana-shim- $py > $rst 
      fi
  done 
}

ana-shim-(){ cat << EOR 

.. include:: ${1:-dummy} 
   :start-after: """
   :end-before: """

EOR
}



