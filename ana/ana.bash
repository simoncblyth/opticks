ana-rel(){      echo ana ; }
ana-src(){      echo ana/ana.bash ; }
ana-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ana-src)} ; }
ana-vi(){       vi $(ana-source) ; }
ana-usage(){ cat << \EOU

ana : Opticks Analysis Scripts
=================================

PMT Tests
------------

:doc:`tpmt`
    Compare Opticks and Geant4 photon bounce histories for simple PMT in box of mineral oil geometry,
    see :doc:`../tests/tpmt`

:doc:`tpmt_distrib`
    Compare Opticks and Geant4 photon distributions for simple PMT in box of mineral oil geometry 

:doc:`tpmt_skimmer`
    Plotting and tabulating mean step by step positions of all photons with specific step histories 
    such as "TO BT BT SA" 


BoxInBox tests
------------------

:doc:`tbox`
    BoxInBox Opticks vs Geant4 history sequence comparisons analogous to **tpmt.py**
    see :doc:`../tests/tbox`

Rainbow Tests
---------------

:doc:`trainbow`
    Rainbow scattering angle comparison between Opticks and Geant4,
    see :doc:`../tests/trainbow`

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

:doc:`twhite`
    Compare wavelength spectrum from :doc:`../tests/twhite` against analytic Planck distribution

**planck.py**
    Planck black body formula

Prism Tests
-------------

**tprism.py**
    Comparison of simulation with analytic expectations for deviation angle vs incident angle,
    see :doc:`../tests/tprism`

**prism_spectrum.py**
    Compare ggv-newton evts with PrismExpected

Reflection Tests
------------------- 

**treflect.py**
    comparison of simulated S and P absolute reflection with Fresnel formula 
    see :doc:`../tests/treflect`

**fresnel.py**
    analytic reflection expectations from Fresnel formula


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

:doc:`tevt`
    Loads single event and dumps constituent array dimensions and photon history tables

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

DEBUGGING SCRIPTS : NOT SURFACING THESE IN THE DOCS
------------------------------------------------------

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
ana-cd(){   cd $(ana-sdir); }

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


ana-pyref-(){ ana-usage | perl -n -e "m,(\w*\.py), && print \"\$1\n\"" | uniq ; }
ana-pyref(){
   ana-cd
   local py
   ana-pyref- | while read py ; do
       [ ! -f "$py" ] && echo $msg bad pyref $py 
       printf " py %30s \n" $py  
   done 
}

ana-docref-(){ ana-usage | perl -n -e "m,:doc:\`(.*)\`, && print \"\$1\n\"" | uniq ; }
ana-docref(){
   ana-cd
   local doc
   local rst
   ana-docref- | while read doc ; do
       rst=$doc.rst
       [ ! -f "$rst" ] && echo $msg BAD DOC REF $rst 
       printf " doc %30s rst %30s \n" $doc $rst  
   done 
}

ana-usage-refcheck-note(){ cat << EON
**ana-usage-refcheck**
    check **ana-usage** for broken py and doc references 
EON
}
ana-usage-refcheck()
{
   ana-pyref
   ana-docref
}


ana-rstcheck()
{
   ana-cd
   local rst
   local py
   grep  -l include:: *.rst | grep -v index.rst | while read rst ; do
       py=${rst/.rst}.py
       [ ! -f "$py" ] && echo $msg dud include $rst   
   done

}



