Tools : bash, python, numpy, ipython, matplotlib
======================================================

Opticks uses bash functions extensively.

Opticks analysis and debugging relies on:

* Python
* NumPy http://www.numpy.org
* IPython
* MatPlotLib



Working with bash functions
-----------------------------

Bash functions make you into a power user of the shell, by
allowing you to operate at a higher level than individual commands
without having to write separate scripts for everything.

.. code-block:: sh 

    delta:tests blyth$ alias t="typeset -f"
    delta:tests blyth$ t opticks-
    opticks- () 
    { 
        source $(opticks-source) && opticks-env $*
    }
    delta:tests blyth$ t tprism-
    tprism- () 
    { 
        . $(opticks-home)/tests/tprism.bash && tprism-env $*
    }
    delta:tests blyth$ t tprism--     # not defined yet 
    delta:tests blyth$ tprism-        # running precursor defines functions including tprism-- tprism-usage and tprism-vi
    delta:tests blyth$ t tprism--
    tprism-- () 
    { 
        local msg="=== $FUNCNAME :";
        local cmdline=$*;
        local pol=${1:-s};
        shift;
        local tag=$(tprism-tag $pol);
        echo $msg pol $pol tag $tag;
        local phifrac0=0.1667;
        local phifrac1=0.4167;
        local phifrac2=0.6667;
        local quadrant=$phifrac1,$phifrac2;
        local critical=0.4854,0.4855;
        local material=GlassSchottF2;
        local azimuth=$quadrant;
        local surfaceNormal=0,1,0;
        local torch_config=(type=invcylinder photons=500000 mode=${pol}pol,wavelengthComb polarization=$surfaceNormal frame=-1 transform=0.500,0.866,0.000,0.000,-0.866,0.500,0.000,0.000,0.000,0.000,1.000,0.000,-86.603,0.000,0.000,1.000 target=0,-500,0 source=0,0,0 radius=300 distance=25 zenithazimuth=0,1,$azimuth material=Vacuum wavelength=0);
        local test_config=(mode=BoxInBox analytic=1 shape=box parameters=-1,1,0,700 boundary=Rock//perfectAbsorbSurface/Vacuum shape=prism parameters=60,300,300,200 boundary=Vacuum///$material);
        op.sh $* --animtimemax 7 --timemax 7 --geocenter --eye 0,0,1 --test --testconfig "$(join _ ${test_config[@]})" --torch --torchconfig "$(join _ ${torch_config[@]})" --torchdbg --save --tag $tag --cat $(tprism-det)
    }

    delta:tests blyth$ tprism-  # pressing [tab] shows available functions beginning tprism-
    tprism-        tprism-args    tprism-det     tprism-env     tprism-py      tprism-src     tprism-test    tprism-vi      
    tprism--       tprism-cd      tprism-dir     tprism-pol     tprism-source  tprism-tag     tprism-usage   
    delta:tests blyth$ tprism-    



IPython : interactive python example
---------------------------------------

IPython enables you to inspect live python objects, so you can 
learn by discovery.  In order to see the plots created by Opticks analysis
scripts you will need to use IPython. 

.. code-block:: py

    delta:ana blyth$ ipython
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks

    In [1]: run tprism.py --tag 1
    tprism.py --tag 1
    INFO:__main__:sel prism/torch/  1 : TO BT BT SA 20160716-1941 /tmp/blyth/opticks/evt/prism/fdomtorch/1.npy 
    INFO:__main__:prism Prism(array([  60.,  300.,  300.,    0.]),Boundary Vacuum///GlassSchottF2 ) alpha 60.0  
    ...







