tg4gun-source(){   echo $(opticks-home)/tests/tg4gun.bash ; }
tg4gun-vi(){       vi $(tg4gun-source) ; }
tg4gun-usage(){ cat << \EOU

tg4gun- : Geant4 Particle Gun Within Dayabay Geometry
=========================================================

`tg4gun--`
     geant4 particle gun simulation within default DYB geometry, loaded from GDML

`tg4gun-v`
     visualize the geant4 propagation

EXERCISE
---------

* change parameters of G4Gun, simulate and visualize.
  For example try a muon, or increase the
  energy and look for more photons.

* develop python script using numpy to check whether aspects 
  of the saved events conform to expectations

Initial configuration of G4Gun:

.. code-block:: sh

    local g4gun_config=(
                 comment=$FUNCNAME
                 particle=e+
                 number=1
                 frame=3153
                 position=0,0,0
                 direction=0,0,1
              polarization=1,0,0
                      time=0.
                    energy=10.0
                   ) 
          # mm, ns, MeV


EOU
}
tg4gun-env(){      olocal- ;  }
tg4gun-dir(){ echo $(opticks-home)/tests ; }
tg4gun-cd(){  cd $(tg4gun-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tg4gun-tag(){ echo 1 ; }
tg4gun-det(){ echo G4Gun ; }
tg4gun-src(){ echo g4gun ; }


tg4gun-args() {        echo  --tag $(tg4gun-tag) --det $(tg4gun-det) --src $(tg4gun-src) ; }
tg4gun-i(){     ipython -i $(which g4gun.py) --  $(tg4gun-args) $* ; }
tg4gun--()
{
    type $FUNCNAME

    local msg="=== $FUNCNAME :"
    local tag=$(tg4gun-tag)

    local g4gun_config=(
                 comment=$FUNCNAME
                 particle=e-
                 number=1
                 frame=3153
                 position=0,0,0
                 direction=1,0,0
              polarization=0,1,0
                      time=0.
                    energy=10.0
                   ) 
          # mm, ns, MeV

   op.sh \
       --okg4 \
       --cat G4Gun --tag $tag --save \
       --g4gun --g4gundbg --g4gunconfig "$(join _ ${g4gun_config[@]})" \
       --rendermode +global,+axis,+in \
       $* 
}


tg4gun-t()
{
    tg4gun-
    tg4gun-- $* --compute
}

tg4gun-v()
{
    tg4gun-
    tg4gun-- --load --optixviz --target 3153 $*
}


