ana-rel(){      echo ana ; }
ana-src(){      echo ana/ana.bash ; }
ana-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ana-src)} ; }
ana-vi(){       vi $(ana-source) ; }
ana-usage(){ cat << EOU

Opticks Analysis
====================




EOU
}

ana-notes(){ cat << EON

CGDMLDetector.py



IndexerTest.py
PropLib.py
__init__.py
ana.py
analytic_cf_triangulated.py
base.py
box_test.py
cfg4_speedplot.py
cfplot.py
cie.py
dae.py
droplet.py
evt.py
fresnel.py
g4gun.py
genstep.py
genstep_sequence_material_mismatch.py
geometry.py
groupvel.py
history.py
material.py
mesh.py
metadata.py
nbase.py
ncensus.py
nload.py
nopstep_viz_debug.py
opticks_env.py
pmt_edge.py
pmt_skimmer.py
pmt_test.py
pmt_test_distrib.py
pmt_test_evt.py
polarization.py
prism.py
prism_spectrum.py
rainbow.py
rainbow_cfg4.py
rainbow_check.py
rainbow_scatter.py
reflection.py
seq.py
source.py
sphere.py
torchevt.py
types.py
utils.py
vacuum_offset.py
xmatlib.py




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





