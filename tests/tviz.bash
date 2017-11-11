tviz-source(){   echo $(opticks-home)/tests/tviz.bash ; }
tviz-vi(){       vi $(tviz-source) ; }
tviz-usage(){ cat << \EOU

tviz- : Visualization Examples
==================================================


`tviz-jpmt-cerenkov`
     Visualize JUNO geometry and photon simulation of Cerenkov and subsequent 
     scintillation light from a 100 GeV muon crossing the JUNO detector.

`tviz-jpmt-scintillation`
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





tviz-jpmt-(){
      op.sh \
           --jpmt \
           --jwire \
           --target 64670 \
           --load \
           --animtimemax 200 \
           --timemax 200 \
           --optixviz \
            $* 
}

#  --fullscreen \

tviz-jpmt-cerenkov(){      tviz-jpmt- --cerenkov $* ; }
tviz-jpmt-scintillation(){ tviz-jpmt- --scintillation $*  ; }





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



