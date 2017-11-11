tviz-source(){   echo $(opticks-home)/tests/tviz.bash ; }
tviz-vi(){       vi $(tviz-source) ; }
tviz-usage(){ cat << \EOU

tviz- : Visualization Examples
==================================================


`tviz-jun-cerenkov`
     Visualize JUNO geometry and photon simulation of Cerenkov and subsequent 
     scintillation light from a 100 GeV muon crossing the JUNO detector.

`tviz-jun-scintillation`
     Visualize JUNO geometry and a photon simulation of scintillation light
     from a 100 GeV muon crossing the JUNO detector.



`tviz-dyb-cerenkov`
     Visualize Dayabay Near site geometry and a photon simulation of 
     Cerenkov and subsequent scintillation light from a 100 GeV muon

`tviz-dyb-scintillation`
     Visualize Dayabay Near site geometry and a photon simulation of scintillation light
     from a 100 GeV muon.


`tviz-dfar`
     Visualize Dayaby Far site geometry 




Note on Analytic GPU geometry
------------------------------

The option *--gltf 1* or *--gltf 3* used by some of 
these commands switches on the use of analytic GPU geometry.

For this to work a prior step to convert GDML to GLTF is needed, eg with:: 

   op --j1707 --gdml2gltf  



EXERCISE
------------

* look at the implementation of the above `tviz-` bash functions,
  run the functions and explore the geometries and event propagations. 
  
For guidance on usage of interactive Opticks see :doc:`../docs/visualization` 



EOU
}
tviz-env(){      olocal- ;  }
tviz-dir(){ echo $(opticks-home)/tests ; }
tviz-cd(){  cd $(tviz-dir); }






tviz-jun-(){
      op.sh \
           --j1707 \
           --gltf 3 \
           --animtimemax 200 \
           --timemax 200 \
           --optixviz \
            $* 
}

#  --fullscreen \

tviz-jun-cerenkov(){      tviz-jun- --cerenkov $* ; }
tviz-jun-scintillation(){ tviz-jun- --scintillation $*  ; }







tviz-dyb-(){
      op.sh \
           --dyb \
           --load \
           --target 3153 \
           --animtimemax 100 \
           --timemax 100 \
           --optixviz \
           --fullscreen \
            $* 
}
tviz-dyb-cerenkov(){      tviz-dyb- --cerenkov $* ; }
tviz-dyb-scintillation(){ tviz-dyb- --scintillation $*  ; }
tviz-dyb-torch(){         tviz-dyb- --torch $*  ; }


tviz-dfar(){
      op.sh \
           --dfar \
           --tracer \
           --fullscreen \
            $* 
}



