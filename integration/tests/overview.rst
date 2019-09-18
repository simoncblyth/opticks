Overview of Opticks Integration Tests 
======================================= 

Sourcing **opticks.bash** results in sourcing of the integration test
precursor bash functions.

.. code-block:: sh

    #### opticks top level tests ########
    
    tviz-(){       . $(opticks-home)/tests/tviz.bash     && tviz-env $* ; }
    tpmt-(){       . $(opticks-home)/tests/tpmt.bash     && tpmt-env $* ; }
    trainbow-(){   . $(opticks-home)/tests/trainbow.bash && trainbow-env $* ; }
    tnewton-(){    . $(opticks-home)/tests/tnewton.bash  && tnewton-env $* ; }
    tprism-(){     . $(opticks-home)/tests/tprism.bash   && tprism-env $* ; }
    tbox-(){       . $(opticks-home)/tests/tbox.bash     && tbox-env $* ; }
    treflect-(){   . $(opticks-home)/tests/treflect.bash && treflect-env $* ; }
    twhite-(){     . $(opticks-home)/tests/twhite.bash   && twhite-env $* ; }
    tlens-(){      . $(opticks-home)/tests/tlens.bash    && tlens-env $* ; }
    tg4gun-(){     . $(opticks-home)/tests/tg4gun.bash   && tg4gun-env $* ; }


Running a precursor function such as `tpmt-` defines other functions 
all beginning `tpmt-` such as `tpmt-vi` and `tpmt-usage`. 
For more about bash function usage see :doc:`../ana/tools`.
    
All the tests have a function such as `tpmt--` or `treflect--` 
which invokes **op.sh** :doc:`../bin/op` with arguments specific to the test
which define the geometry, light source (or G4gun parameters) and 
where to save event files. 

.. code-block:: sh

   join(){ local IFS="$1"; shift; echo "$*"; }
   op.sh \
       --cat PmtInBox --tag 10 --save \
       --test --testconfig "$(join _ ${test_config[@]})" \
       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --timemax 10 \
       --animtimemax 10 \
       --eye 0.0,-0.5,0.0 \
       --geocenter \
       $*


* **--cat** : category of the event, for test geometries this
  overrides the detector name 

* **--tag** : numerical identifier of a simulation setup within a category, 
  for example S-polarized and P-polarized setups use different tags.
  Also the convention is adopted of giving a Geant4 equivalent setup
  a negative tag of the same magnitude.

* **--save** : saves event files, by default into folders within /tmp/$USER/opticks/evt/

* **--test** : swiches on use of a test geometry, rather than a full detector geometry

* **--testconfig** : configures the test geometry, described below

* **--torch** : swiches on use of artificial light source instead of **--cerenkov** or **--scintillation**

* **--torchconfig** : configures the light source, described below

* **--timemax** : maximum time in nanoseconds for the simulation, used to set the time compression domain
 

The below **join** functions simply concatenate the **torch_config** and **test_config** 
arrays into strings delimited by **_** to simplify passing the config as commandline 
options to the Opticks executables.

.. code-block:: sh 

       --torch --torchconfig "$(join _ ${torch_config[@]})" \
       --test --testconfig "$(join _ ${test_config[@]})" \


Test Geometry Configuration
-------------------------------

The code that interprets the configuration is :oktip:`ggeo/GGeoTestConfig.cc`.

.. code-block:: sh

     local material=GlassSchottF2
     local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=box   parameters=-1,1,0,700           boundary=Rock//perfectAbsorbSurface/Vacuum
                 shape=lens  parameters=641.2,641.2,-600,600 boundary=Vacuum///$material
               )


Multiple **shape**, **parameters** and **boundary** are supported, there must be 
an equal number of each. Available modes are: **BoxInBox** and **PmtInBox**. 

Test Geometry Shapes
~~~~~~~~~~~~~~~~~~~~~~~

To see the supported shapes and discover the meaning of the parameters
start from :oktip:`ggeo/GMaker.cc`.


Boundary Specification
~~~~~~~~~~~~~~~~~~~~~~~~

Boundaries are composed of four parts:

* outer material
* outer surface, relevant to incoming photons
* inner surface, relevant to outgoing photons
* inner material

They are specified by a string with **/** delimiters, 
surfaces are optional, materials are required.

.. code-block:: sh

    Rock//perfectAbsorbSurface/Vacuum

See :doc:`../ana/boundary` for more.


Materials
~~~~~~~~~~~

To dump materials names and properties, use `op --mat` eg:

.. code-block:: sh 

    op --mat --ggeo warn  # titles and property names
    op --mat 0            # table of properties of material with index 0
    op --mat GdDopedLS    # table of properties of material identified by name 
    op --mat              # tables of all material properties  


Surfaces
~~~~~~~~~

Similary for dumping surfaces use `op --surf`:
    
.. code-block:: sh 

    op --surf --ggeo warn

Note that surfaces have not yet been debugged, you are advised 
to only use the **perfect** surfaces::

     perfectDetectSurface  
     perfectAbsorbSurface  
     perfectSpecularSurface  
     perfectDiffuseSurface


Torch Configuration
--------------------

Relevant sources:

* :oktip:`opticksnpy/TorchStepNPY.cpp` parses the config
* :oktip:`opticksnpy/TorchStepNPY.hpp` header used from both C++ and CUDA
* :oktip:`optixrap/cu/torchstep.h` 
* :doc:`../optixrap/cu/torchstep`

Example torch config used by :doc:`treflect`

.. code-block:: sh 

    local pol=s
    local photons=1000000
    local torch_config=(
                 type=refltest
                 photons=$photons
                 mode=${pol}pol,flatTheta
                 polarization=0,0,-1
                 frame=-1
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000
                 source=0,0,-200
                 radius=100
                 distance=25
                 zenithazimuth=0.5,1,0,1
                 material=Vacuum
                 wavelength=550
               )



Updating Opticks Bash Functions
---------------------------------

If you have not yet cloned Opticks to your machine, do so with::

     cd
     hg clone http://bitbucket.org/simoncblyth/opticks  

If you have already done the above, make sure to update to the latest version with::

     cd ~/opticks    
     hg pull
     hg update

Then update your bash enviroment by running::

     opticks-



Analysis Only Exercises
-------------------------

Sections labelled **ANALYSIS EXERCISE** can be done without a full Opticks installation.
You will however need to copy some files to your machine, using  **opticks-** bash functions.

First update you Opticks bash functions as described above
and then run the below function to copy files to your machine::

     opticks-analysis-only-setup 

To see what the function does use the **t** alias that 
you should have in your `~/.bash_profile` ::

    [simonblyth@optix ~]$ t opticks-analysis-only-setup  # alias t="typeset -f"
    opticks-analysis-only-setup () 
    { 
        opticksdata-;
        opticksdata-get;
        opticks-geo-get;
        opticks-evt-get rainbow
    }

This will **scp** files from the workstation installation, you will
be prompted for your workstation password several times.  


Note this only copies **rainbow** files, to do other exercises you will
need to run for example::

    opticks-evt-get reflect



 

    


 





