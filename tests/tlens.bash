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


Ideas
-------

Currently there is a frailty from the separate specification
of test geometry dimensions and test photon source positions,
from the implicit coupling.  

* decouple how ? how to pin source onto geometry 





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

  
TIPS
------

* when making analysis only code changes, can load event and NCSG geometry and run analysis
  with tlens-ana no need to rerun simulation every time 


EOU
}
tlens-env(){      olocal- ;  }
tlens-dir(){ echo $(opticks-home)/tests ; }
tlens-cd(){  cd $(tlens-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tlens-det(){ echo lens ; }
tlens-src(){ echo torch ; }

tlens-args() {        echo  --det $(tlens-det) --src $(tlens-src) ; }


tlens-medium(){ echo Vacuum ; }
tlens-container(){ echo Rock//perfectAbsorbSurface/$(tlens-medium) ; }
tlens-testobject(){ echo Vacuum///GlassSchottF2 ; }

tlens-ana-(){  
    local msg="$FUNCNAME :"
    #local dbgseqhis=0x8cbc6d   # TO SC BT BR BT SA 
    local dbgseqhis=0x8cc6d   # TO SC BT BT SA
    #local dbgseqhis=0x8ccd   # TO BT BT SA

    local testname=${TESTNAME}
    [ -z "$testname" ] && echo $msg missing TESTNAME && sleep 1000000 

    OpticksEventAnaTest --torch  --tag 1 --cat $testname  --dbgnode 0  --dbgseqhis $dbgseqhis ; 
}

tlens-pload(){ tevt.py   $(tlens-args) --tag 1  ; }
tlens-py() {   tlens.py  $(tlens-args) $* ; } 
tlens-ipy() {  ipython -i -- $(which tlens.py)  $(tlens-args) $* ; } 
 


tlens-convex-a(){ TESTNAME=${FUNCNAME/-a} tlens-ana- ; }
tlens-convex(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tlens-- s $* ; }
tlens-convex-(){ $FUNCNAME- | python $* ; }  
tlens-convex--(){ cat << EOP 

from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgpath="$TMP/$FUNCNAME", testobject="$(tlens-testobject)", container="$(tlens-container)" )

container = CSG("box", param=[-1,1,0,700], boundary=args.container, poly="MC", nx="20" )

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution="40", verbosity="0", ctrl="0" )

al = CSG("sphere", param=[0,0,-600,641.2])   
ar = CSG("sphere", param=[0,0, 600,641.2])
lens = CSG("intersection", left=al, right=ar )

CSG.Serialize([container, lens ], args.csgpath )

EOP
}


tlens-concave-a(){ TESTNAME=${FUNCNAME/-a} tlens-ana- ; }
tlens-concave(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tlens-- s $* ; }
tlens-concave-(){ $FUNCNAME- | python $* ; }  
tlens-concave--(){ cat << EOP 

import logging 
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  

args = opticks_main(csgpath="$TMP/$FUNCNAME", testobject="$(tlens-testobject)", container="$(tlens-container)" )

cr = 300.
cz = 30.

sz = (cz*cz + cr*cr)/(2.*cz )
sr = sz

log.info( " cr %s cz %s sr %s sz %s " % (cr,cz,sr,sz ))


container = CSG("box", param=[0,0,0,2*sz], boundary=args.container, poly="MC", nx="20" )
log.info(" container.boundary : %s " % container.boundary )

CSG.boundary = args.testobject
CSG.kwa = dict(poly="IM", resolution="50", verbosity="0", ctrl="0" )

cy = CSG("cylinder", param=[0,0,0,cr], param1=[-cz,cz,0,0])   
bx = CSG("box3",     param=[cr,cr,cz,0] )   


ar = CSG("sphere", param=[0,0, sz,sr], complement=False)
al = CSG("sphere", param=[0,0,-sz,sr], complement=False)

delta = 0.1

# z1,z2 relative to zsphere center (?)

br = CSG("zsphere", param=[0,0, sz,sr], param1=[-sr         , -sr+cz+delta,0,0], complement=False)
bl = CSG("zsphere", param=[0,0,-sz,sr], param1=[ sr-cz-delta,     sr,0,0], complement=False)


#lens = cy - ar - al 
lens = cy - br - bl 
#lens = bx - br - bl 
#lens = cy - al 
#lens = cy  
#lens = al  

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
                 source=0,0,-300
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
    local tag
    case $pol in  
        s) tag=1 ;;
        p) tag=2 ;;
        *) echo "tlens-- expects s/p argument " && sleep 100000000 ;;
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

    local testname
    if [ -n "$TESTNAME" ]; then
        testname=${TESTNAME}
    else
        testname=$(tlens-det)
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
            --save --tag $tag --cat $testname \
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
