tlens-source(){   echo $(opticks-home)/tests/tlens.bash ; }
tlens-vi(){       vi $(tlens-source) ; }
tlens-usage(){ cat << \EOU

tlens- : Disc shaped beam of white light incident on convex lens  
====================================================================


`tlens-vi`
    edit the bash functions 

`tlens--`
    create Opticks geometry, simulates photons in interop mode, visualize, saves evt file 

`tlens-convex`
    works as expected, qualitatively

`tlens-concave`
    NOT WORKING : photons unmoved, boundary issue ?





Theory
---------

Spherical lens formula

* http://www.physicsinsights.org/simple_optics_spherical_lenses-1.html



EXERCISE
-----------

* Change the lens material and interpret what you get, 
  see :doc:`overview` regarding materials.

* Try adding one or more lens, for example with line::

     shape=lens  parameters=641.2,641.2,-400,800 boundary=Vacuum///$material

* Try adding a different shape, examine the **GMaker** source code :oktip:`ggeo/GMaker.cc`
  to see what shapes are available

* Write a python analysis script **tlens.py** that 
  
    * loads an event from `tlens-`
    * prints the photon history table
    * select a subset of the photons (using **seqs** argument to Evt class)
    * plot distributions for the subset using **matplotlib**  
    * interpret the plot

  
 


EOU
}
tlens-env(){      olocal- ;  }
tlens-dir(){ echo $(opticks-home)/tests ; }
tlens-cd(){  cd $(tlens-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tlens-det(){ echo lens ; }
tlens-src(){ echo torch ; }

tlens-args() {        echo  --det $(tlens-det) --src $(tlens-src) ; }
tlens-py() {          tlens.py  $(tlens-args) $* ; } 


tlens-medium(){ echo Vacuum ; }
tlens-container(){ echo Rock//perfectAbsorbSurface/$(tlens-medium) ; }
tlens-testobject(){ echo Vacuum///GlassSchottF2 ; }


tlens-convex(){ TESTCONFIG=$($FUNCNAME- 2>/dev/null) tlens-- $* ; }
tlens-convex-(){ $FUNCNAME- | python $* ; }  
tlens-convex--(){ cat << EOP 

from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgpath="$TMP/$FUNCNAME", testobject="$(tlens-testobject)", container="$(tlens-container)" )

container = CSG("box", param=[-1,1,0,700], boundary=args.container, poly="MC", nx="20" )

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution="40", verbosity="1", ctrl="0" )

al = CSG("sphere", param=[0,0,-600,641.2])   
ar = CSG("sphere", param=[0,0, 600,641.2])
lens = CSG("intersection", left=al, right=ar )

CSG.Serialize([container, lens ], args.csgpath )

EOP
}


tlens-concave(){ TESTCONFIG=$($FUNCNAME- 2>/dev/null) tlens-- $* ; }
tlens-concave-(){ $FUNCNAME- | python $* ; }  
tlens-concave--(){ cat << EOP 

import logging 
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgpath="$TMP/$FUNCNAME", testobject="$(tlens-testobject)", container="$(tlens-container)" )

cr = 300.
cz = 100.

sz = (cz*cz + cr*cr)/(2.*cz )
sr = sz

log.info( " cr %s cz %s sr %s sz %s " % (cr,cz,sr,sz ))


container = CSG("box", param=[0,0,0,sz], boundary=args.container, poly="MC", nx="20" )
log.info(" container.boundary : %s " % container.boundary )

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution="50", verbosity="1", ctrl="0" )

cy = CSG("cylinder", param=[0,0,0,cr], param1=[-cz,cz,0,0])   
ar = CSG("sphere", param=[0,0, sz,sr], complement=False)
al = CSG("sphere", param=[0,0,-sz,sr], complement=False)

lens = cy - ar - al 

#la = CSG("intersection", left=cy, right=ar )

log.info(" lens.boundary : %s " % lens.boundary )


CSG.Serialize([container, lens ], args.csgpath )

"""

          (-cz,cr)      (+cz,cr)
               +---------+ 
               |    |    |
               |    |    |
               |    |    |
               |    |    |                                
      ---------|----0----|-----------+------------------   --> Z
               |    |    |         (sz
               |    |    |
               |    |    |
               |    |    |
               +---------+ 

    Find parameters of sphere that goes thru points (0,0) and (cz,cr)

       sz = sr 

      (sz - cz)^2 + cr^2 = sr^2

        sz^2 - 2 sz cz + cz^2 + cr^2 = sz^2
 
                sz = (cz^2 + cr^2)/(2*cz)


"""

EOP
}



tlens-torchconfig()
{
    local pol=$1
    local torch_config=(
                 type=disc
                 photons=500000
                 mode=${pol}pol,wavelengthSource
                 frame=-1
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000
                 target=0,0,0
                 source=0,0,-600
                 time=0.1
                 radius=100
                 distance=500
                 zenithazimuth=0,1,0,1
                 material=$(tlens-medium)
               )

     echo "$(join _ ${torch_config[@]})" 
}


tlens--()
{
    type $FUNCNAME
    local pol=${1:-s}
    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
    esac
    echo  pol $pol tag $tag


    local testconfig
    if [ -n "$TESTCONFIG" ]; then
        testconfig=${TESTCONFIG}
    else
        testconfig=$(tlens-convex-)
    fi

    local torchconfig
    if [ -n "$TORCHCONFIG" ]; then
        torchconfig=${TORCHCONFIG}
    else
        torchconfig=$(tlens-torchconfig $pol)
    fi


    op.sh  \
            $* \
            --animtimemax 7 \
            --timemax 7 \
            --geocenter \
            --eye 0,1,0 \
            --up  1,0,0 \
            --test --testconfig "$testconfig" \
            --torch --torchconfig "$torchconfig" \
            --torchdbg \
            --save --tag $tag --cat $(tlens-det) \
            --rendermode +global,+axis
}

tlens-t()
{
    tlens-- s --compute
    tlens-- p --compute
}

tlens-v()
{
    tlens-- ${1:-s} --load
}
