Tools : bash, python, numpy, ipython, matplotlib
======================================================

Opticks uses bash functions extensively and 
analysis and debugging relies on:

* Python
* IPython
* NumPy http://www.numpy.org
* MatPlotLib


In general the best way to install common software such as
these tools is with your systems package manager. For example:

* Linux: yum, apt-get
* macOS: https://www.macports.org or homebrew
* Windows: https://chocolatey.org 

However there are other ways to install packages such as
using **pip** for python packages.


IPython Install on a mac with pip
----------------------------------

Installing ipython on a mac if you do not use macports.

::

    delta:tests blyth$ sudo pip install ipython==1.2.1
    Password:
    Downloading/unpacking ipython==1.2.1
      Downloading ipython-1.2.1.tar.gz (8.7MB): 8.7MB downloaded
      Running setup.py (path:/private/tmp/pip_build_root/ipython/setup.py) egg_info for package ipython
        running egg_info
        
    Installing collected packages: ipython
      Running setup.py install for ipython
        running install
        
        Installing iplogger script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing iptest script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing ipcluster script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing ipython script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing pycolor script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing ipcontroller script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing irunner script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
        Installing ipengine script to /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
    Successfully installed ipython
    Cleaning up...
    delta:tests blyth$ 
    delta:tests blyth$ which ipython
    /opt/local/bin/ipython
    delta:tests blyth$ ipython
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: import numpy as np

    In [2]: np.__version__
    Out[2]: '1.9.2'

    In [3]: import matplotlib

    In [4]: import matplotlib.pyplot as plt 

    In [5]: matplotlib.__version__
    Out[5]: '1.3.1'





Working with bash functions
-----------------------------

Bash functions make you into a power user of the shell, by
allowing you to operate at a higher level than individual commands
without having to write separate scripts for everything.

.. code-block:: sh 

    delta:tests blyth$ alias t="typeset -f"    # enables introspection of bash functions
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



A powerful way to rapidly develop bash functions is to use 
two terminal sessions, in the first edit the functions:

.. code-block:: sh

    tprism-vi  

In the second test the **tprism-foo** you are developing:

.. code-block:: sh

    tprism-;tprism-foo    # tprism- updates function definitions and the tprism-foo runs it 



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


IPython Tab Completion
~~~~~~~~~~~~~~~~~~~~~~~~

Discover available methods of an object by interactive exploration:

.. code-block:: py

    In [2]: evt.  # press [tab] to see the possibilities
    evt.RPOL                evt.desc                evt.histype             evt.material_table      evt.ph                  evt.rflgs_              evt.seqhis              evt.td
    evt.RPOST               evt.description         evt.idom                evt.mattype             evt.polw                evt.rpol_               evt.seqhis_or_not       evt.tdii
    evt.RQWN_BINSCALE       evt.det                 evt.incident_angle      evt.msize               evt.post                evt.rpol_bins           evt.seqmat              evt.unique_wavelength
    evt.a_deviation_angle   evt.deviation_angle     evt.init_index          evt.nrec                evt.post_center_extent  evt.rpost_              evt.seqs                evt.valid
    evt.a_recside           evt.dirw                evt.init_metadata       evt.ox                  evt.ps                  evt.rs                  evt.src                 evt.wl
    evt.a_side              evt.fdom                evt.init_photons        evt.p0                  evt.py                  evt.rsmry_              evt.stamp               evt.x
    evt.all_history         evt.flags               evt.init_records        evt.p_out               evt.pyc                 evt.rsr                 evt.summary             evt.y
    evt.all_material        evt.flags_table         evt.init_selection      evt.path                evt.rec                 evt.rx                  evt.t                   evt.z
    evt.brief               evt.history             evt.label               evt.paths               evt.recflags            evt.rx_raw              evt.tag                 evt.zrt_profile
    evt.c4                  evt.history_table       evt.material            evt.pbins               evt.recwavelength       evt.selection           evt.tbins               



One question mark gives the documentation:

.. code-block:: py 

    In [2]: evt.deviation_angle?
    Type:       instancemethod
    File:       /Users/blyth/opticks/ana/evt.py
    Definition: evt.deviation_angle(self, side=None, incident=None)
    Docstring:
    Deviation angle for parallel squadrons of incident photons 
    without assuming a bounce count

Two question marks gives the implementation:

.. code-block:: py 

    In [4]: evt.zrt_profile??

    Type:       instancemethod
    File:       /Users/blyth/opticks/ana/evt.py
    Definition: evt.zrt_profile(self, n, pol=True)
    Source:
        def zrt_profile(self, n, pol=True):
            """
            :param n: number of bounce steps 
            :return: min, max, mid triplets for z, r and t  at n bounce steps

            ::

                In [7]: a_zrt
                Out[7]: 
                array([[ 300.    ,  300.    ,  300.    ,    1.1748,   97.0913,   49.133 ,    0.1001,    0.1001,    0.1001],
                       [  74.2698,  130.9977,  102.6337,    1.1748,   97.0913,   49.133 ,    0.9357,    1.2165,    1.0761],
                       [  56.0045,  127.9946,   91.9996,    1.1748,   98.1444,   49.6596,    0.9503,    1.3053,    1.1278]])


            """
            slab = "z r t"
            if pol:
                slab += " lx ly lz"

            labs = slab.split()
            nqwn = 3
            zrt = np.zeros((n,len(labs)*nqwn))
            tfmt = "%10.3f " * nqwn
            fmt = " ".join(["%s: %s " % (lab, tfmt) for lab in labs])

            for i in range(n):
                p = self.rpost_(i)
                l = self.rpol_(i)
                lx = l[:,0]
                ly = l[:,1]
                lz = l[:,2]







NumPy
-------

Opticks although mostly implemented in C++ uses the NPY serialization format
for all geometry and event buffers allowing debugging and analysis
to be done from the IPython using NumPy.

NPY serialized files used extension **.npy**, find some:

.. code-block:: sh

    delta:~ blyth$ find /tmp/blyth/opticks/evt/lens -name 1.npy
    /tmp/blyth/opticks/evt/lens/fdomtorch/1.npy
    /tmp/blyth/opticks/evt/lens/idomtorch/1.npy
    /tmp/blyth/opticks/evt/lens/notorch/1.npy
    /tmp/blyth/opticks/evt/lens/oxtorch/1.npy
    /tmp/blyth/opticks/evt/lens/phtorch/1.npy
    /tmp/blyth/opticks/evt/lens/pstorch/1.npy
    /tmp/blyth/opticks/evt/lens/rstorch/1.npy
    /tmp/blyth/opticks/evt/lens/rxtorch/1.npy


Determine maximum photon time **in one line**:

.. code-block:: sh

    delta:~ blyth$ python -c "import numpy as np ; print np.load('/tmp/blyth/opticks/evt/lens/oxtorch/1.npy')[:,0,3].max() "
    11.2614

Do that more sedately with ipython:

.. code-block:: py

    delta:~ blyth$ ipython
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks

    In [1]: ox = np.load("/tmp/blyth/opticks/evt/lens/oxtorch/1.npy")

    In [2]: ox
    Out[2]: 
    array([[[   1.884,    0.202,  700.   ,    4.569],
            [   0.098,    0.011,    0.995,    1.   ],
            [  -0.107,    0.994,    0.   ,  630.652],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.745,   -1.288,  700.   ,    4.568],
            [   0.05 ,   -0.087,    0.995,    1.   ],
            [   0.866,    0.501,    0.   ,  753.122],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ 290.308, -500.146, -700.   ,    6.057],
            [   0.21 ,   -0.433,   -0.877,    1.   ],
            [   0.501,    0.818,   -0.283,  425.268],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           ..., 
           [[   2.217,    0.106,  700.   ,    4.572],
            [   0.078,    0.004,    0.997,    1.   ],
            [  -0.048,    0.999,    0.   ,  472.495],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  -1.872,   -0.614,  700.   ,    4.574],
            [  -0.055,   -0.018,    0.998,    1.   ],
            [   0.312,   -0.95 ,    0.   ,  405.361],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   2.042,   -3.525,  700.   ,    4.571],
            [   0.067,   -0.116,    0.991,    1.   ],
            [   0.865,    0.501,    0.   ,  589.317],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [3]: ox.shape
    Out[3]: (500000, 4, 4)    # 0.5M photons with (4,4) floats each 

    In [4]: ox[0]             # 1st photon 
    Out[4]: 
    array([[   1.884,    0.202,  700.   ,    4.569],
           [   0.098,    0.011,    0.995,    1.   ],
           [  -0.107,    0.994,    0.   ,  630.652],
           [   0.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [5]: ox[-1]           # last photon  
    Out[5]: 
    array([[   2.042,   -3.525,  700.   ,    4.571],
           [   0.067,   -0.116,    0.991,    1.   ],
           [   0.865,    0.501,    0.   ,  589.317],
           [   0.   ,    0.   ,    0.   ,    0.   ]], dtype=float32)

    In [6]: ox[:,0,3]        # "wildcard" first dimension with ":" and pick the other two, giving end times of the photons  
    Out[6]: array([ 4.569,  4.568,  6.057, ...,  4.572,  4.574,  4.571], dtype=float32)

    In [7]: ox[:,0,3].min()  # earliest time 
    Out[7]: 0.15406403

    In [8]: ox[:,0,3].max()  # latest time
    Out[8]: 11.261424


See :doc:`../ana/evt` for a higher level way of loading event buffers.





