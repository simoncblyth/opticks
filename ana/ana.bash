ana-rel(){      echo ana ; }
ana-src(){      echo ana/ana.bash ; }
ana-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ana-src)} ; }
ana-vi(){       vi $(ana-source) ; }
ana-usage(){ cat << \EOU

ana : Opticks Analysis Scripts
=================================


Issue : analysis machinery expects a geocache
---------------------------------------------------

::

    simon:analytic blyth$ ip
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    ...
    IPython profile: g4opticks
    args: /opt/local/bin/ipython --profile=g4opticks
    Invalid/missing IDPATH envvar /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae 
    An exception has occurred, use %tb to see the full traceback.

    SystemExit: 1

    To exit: use 'exit', 'quit', or Ctrl-D.

    In [1]: 



FUNCTION `ana-t`
------------------

After running `tests-t` all the below "MISSING EVT" should become "OK"::

    simon:ana blyth$ ana-t
    ==                    __init__.py ==  ->    0 OK  
    ==                         ana.py ==  ->  101 MISSING EVT  
    ==                        base.py ==  ->    0 OK  
    ==                    boundary.py ==  ->    0 OK  
    ==              cfg4_speedplot.py ==  ->    0 OK  
    ==                      cfplot.py ==  ->    0 OK  
    ==                         cie.py ==  SKIP
    ==                     dbgseed.py ==  ->  101 MISSING EVT  
    ==                     droplet.py ==  ->    0 OK  
    ==                         evt.py ==  ->    0 OK  
    ==                     fresnel.py ==  ->    0 OK  
    ==                       g4gun.py ==  ->  101 MISSING EVT  
    ==                     genstep.py ==  ->    0 OK  
    ==                    geometry.py ==  ->    0 OK  
    ==                     histype.py ==  ->    0 OK  
    ==                    material.py ==  ->    0 OK  
    ==                     mattype.py ==  ->    0 OK  
    ==                  mergedmesh.py ==  ->    0 OK  
    ==                    metadata.py ==  ->    0 OK  
    ==                       nbase.py ==  ->    0 OK  
    ==                     ncensus.py ==  ->    0 OK  
    ==                       nload.py ==  ->    0 OK  
    ==                          ox.py ==  ->    0 OK  
    ==                      planck.py ==  ->    0 OK  
    ==              prism_spectrum.py ==  ->  101 MISSING EVT  
    ==                     proplib.py ==  ->    0 OK  
    ==                         seq.py ==  ->    0 OK  
    ==                      sphere.py ==  ->  101 MISSING EVT  
    ==                        tbox.py ==  ->  101 MISSING EVT  
    ==                        tevt.py ==  ->  101 MISSING EVT  
    ==                       tmeta.py ==  ->    0 OK  
    ==                        tpmt.py ==  ->  101 MISSING EVT  
    ==                tpmt_distrib.py ==  ->  101 MISSING EVT  
    ==                tpmt_skimmer.py ==  ->  101 MISSING EVT  
    ==                      tprism.py ==  ->  101 MISSING EVT  
    ==                    trainbow.py ==  ->  101 MISSING EVT  
    ==                    treflect.py ==  ->  101 MISSING EVT  
    ==                      twhite.py ==  ->  101 MISSING EVT  
    ==                    xrainbow.py ==  ->    0 OK  



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

**tboxlaser.py**
    Looks like simple event comparison, without using AB/CF etc..
    TODO: use the generic AB machinery to do the same


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


Scatter Tests
---------------

**rayleigh.py**
    Without selection scatter distrib plots from arrays created by:

    optixrap/tests/ORayleighTest.cc 
    cfg4/tests/OpRayleighTest.cc


Source Tests
--------------

:doc:`twhite`
    Compare wavelength spectrum from :doc:`../tests/twhite` against analytic Planck distribution

**planck.py**
    Planck black body formula : CDF technique dev, plotting


Full Geometry Tests
----------------------

**tdefault.py**
     tests within default full geometry, using CF (older than the AB/CFH machinery?)

**tdefault_distrib.py**
     tests within default full geometry, using cf,cfplot

**tlaser.py**
     okg4 comparisons manually using Evt (not the newer AB)

**tlaser_distrib.py**
     okg4 comparisons using CF (not the newer CFH)


Analytic/NCSG Geometry tests
-----------------------------

**analytic_boundary_dbg.py**
    analysis of photon_buffer written by oxrap/cu/generate.cu:tracetest

    TODO: rename, too special case to share name with tboolean-   

**tgltf.py** 
    Shakedown analytic geometry, with analytic geometry transform access

Concentric Tests : idealised spherical DYB AD
----------------------------------------------

**tconcentric.py**
     comparison using AB, with scatter polarization debug/plotting
     see tconcentric-i

**tconcentric_distrib.py**
     comparison histo mass production with 
     see tconcentric-d


Lens/Prism Tests
------------------

**tlens.py**
    Single Evt dumping, no comparisons

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

**bnd.py**
    testing PropLib usage to access material and surface properties
 
:doc:`mergedmesh`
    Access geometrical data such as positions, transforms of volumes of the geometry

**geometry.py**
    Shape, Ray, Plane, Intersect, IntersectFrame : simple intersect calulations



Event Comparison Infrastructure
-----------------------------------

**tokg4.py**
    loads evt pair and compares at Evt level

**ab.py**
    event pair comparison with AB class. Used by cfh.py, cfplot.py, tconcentric.py, tconcentric_distrib.py

**abstat.py**
    history table slicing, selection. Used by ab.py, cfh.py

**make_rst_table.py**
    pure python utility used by ab.py abstat.py 

**cf.py**
    Sliced history CF spawning.      
    Used by tdefault.py tdefault_distrib.py tlaser_distrib.py tpmt_distrib.py

**decompression.py**
    Deco : utils for re-binning domain compressed values that avoid binning artifacts

    Used from : ab.py cf.py 

  
**cfh.py**
    CFH : persistable comparison histograms and chi2, with random access via path specification

    ipython -i $(which cfh.py) -- /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X

    Used from : ab.py cf.py cfhload.py cfplot.py tconcentric.py 

**cfhload.py**
    Testing cfh.py CFH

**ctx.py**
    Ctx : Utility providing conversions between various ways of addressing comparison histograms

    Used from : ab.py cfh.py evt.py


Event Infrastructure
-----------------------

:doc:`tevt`
    Loads single event and dumps constituent array dimensions and photon history tables

**evt.py**
    loads event data : used by many ~30 other scripts

**tmeta.py**
    loads and dumps a single events metadata 

    tmeta.py --det tboolean-torus --tag 1


**base.py** 
    internal envvar setup based on input envvar IDPATH 
    json and ini loading with Abbrev and ItemList classes 

**ok.py** 
    test of opticks_main arg parsing/defaults  

**enum.py** 
    parsing C++ enum into python dict. Used by base.py 
 
**nload.py**
    numpy array and .ini metadata loading with clases A, I, II

**ox.py**
    load just final photons, testing nload and arg parsing  

**ph.py**
    load ph photon seqhis, dumping progressing SeqAna sequences


**nbase.py**
    pure numpy utility functions: count_unique, count_unique_sorted, chi2, decompression_bins 

**seq.py**
    SeqType conversions of long integer sequence codes to abbreviation string sequences like  "TO BT BR BR BT SA"
    SeqTable presenting frequencies of different sequences 

    Used by : evt.py hismask.py histype.py mattype.py

**histype.py**
    HisType (SeqType subclass) and tests of history table formation and dumping.
    Used by ab.py cf.py evt.py seq.py. With entry point::

        histype.py --det tboolean-torus --tag -1 --src torch 

**hismask.py**
    converts seqhis into a mask, not the usual sequence : used by evt.py pflags.py

        
**mattype.py**
    MatType (SeqType subclass) and tests

**genstep.py** 
    fit genstep xyz vs time, to obtain parametric eqn for the viewpoint tracking 
    used to create videos, see `vids-`

**genstep_merge.py** 
    combine Cerenkov and scintillation gensteps in natural ones


**ana.py**
    geometrical and plotting utils

**OpticksQuery.py**
    analog of okc/OpticksQuery.cc geometry selection string parsing available from opticks_main args.query 



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
    event census looking across many events with array shape dumping 


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

mm_buffer.py 
     debugging mesh buffer structure


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

ana.py
     misc old analysis funcs. Used from tprism.py, treflect.py

cf5d.py
     5d Proplib comparison (dev code ?)

dat.py 
     Dat : For interactive exploration of dimension > 3  arrays

dbgseed.py
     old dev script used from oks-dbgseed

gltf.py
     dev exploration of precision of transforms parsed from GLTF geometry json files

gs.py
     example of pure numpy dumping of opticks buffers

nodelib.py 
     check analytic and triangulates LVNames, PVNames lists 

old_cfplot.py
     looks ancient

pflags.py
     debugging inconsistent pflags in CFG4 evt

seqmat.py
     debugging seqmat mismatch, zeros.

truncation.py
     pseudo duplication of oxrap/cu/generate.cu for easy thinking about truncation
     and thinking about seqhis steered propagations 



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

ana-t-note(){ cat << EON
*ana-t*
     runs all executable .py in ana directory, 
     stopping at any script that yields a return code 
     that is not zero or 101

     Return code 101 is used to signal missing simulated evts 

     Use to check for simple python breakages from moving 
     things around for example. 

     Intended to just test mechanics, not meaning.
     Failure to load events should not cause failures, just warnings.
EON
}

ana-t-rcm()
{
   case $1 in 
     0)  echo OK ;;
   101)  echo MISSING EVT ;;  
     *)  echo PY FAIL ;;
   esac
}

ana-t(){
   local py
   local rc
   local rcm
   ana-cd
   ana-py- | while read py ; do

       printf "== %30s == " $msg $py

       [ ! -x $py ] && printf " SKIP\n" && continue

       ./$py > /dev/null 2>&1
       rc=$?
       rcm="$(ana-t-rcm $rc)"
       printf " -> %4s %s  \n" $rc "$rcm"

       if [ "$rc" == "0" -o "$rc" == "101" ]; then 
           echo -n
       else
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




ana-mod-ls-()
{  
   ana-cd 
   ls -1 *.py
}

ana-mod-from()
{
   local module
   ana-mod-ls- | while read module 
   do
       ana-mod-from- ${module/.py} 
   done
}

ana-mod-from-()
{
   local msg="=== $FUNCNAME : "
   local module=${1:-cf}
   #echo
   echo $msg module $module used from :
   ana-cd
   $FUNCNAME- $module
}

ana-mod-from--()
{
    grep -l from\ opticks.ana.${1:-cf}\  *.py 
}
