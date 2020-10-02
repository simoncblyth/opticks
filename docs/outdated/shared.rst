Using a Shared Opticks Installation
====================================

If someone has installed Opticks for you already 
you just need to set the PATH variable in your .bash_profile 
to easily find the Opticks executables and scripts. 

.. code-block:: sh

    # .bash_profile

    # Get the aliases and functions
    if [ -f ~/.bashrc ]; then
        . ~/.bashrc
    fi

    # User specific environment and startup programs

    PATH=$PATH:$HOME/.local/bin:$HOME/bin
    ini(){ . ~/.bash_profile ; }

    ok-local(){    echo /home/simonblyth/local ; }
    ok-opticks(){  echo /home/simonblyth/opticks ; }
    ok-ctest(){    ( cd $(ok-local)/opticks/build ; ctest3 $* ; ) }

    export PATH=$(ok-opticks)/ana:$(ok-opticks)/bin:$(ok-local)/opticks/lib:$PATH


You can test the installation using the `ok-ctest` function defined in 
the .bash_profile. The output shoule look like the below. 
The permission denied error is not a problem.

.. code-block:: sh

    [blyth@optix ~]$ ok-ctest
    Test project /home/simonblyth/local/opticks/build
    CMake Error: Cannot open file for write: /home/simonblyth/local/opticks/build/Testing/Temporary/LastTest.log.tmp
    CMake Error: : System Error: Permission denied
    Problem opening file: /home/simonblyth/local/opticks/build/Testing/Temporary/LastTest.log
    Cannot create log file: LastTest.log
            Start   1: SysRapTest.SEnvTest
      1/155 Test   #1: SysRapTest.SEnvTest ........................   Passed    0.00 sec
            Start   2: SysRapTest.SSysTest
    ...
    ...
    154/155 Test #154: cfg4Test.G4StringTest ......................   Passed    0.06 sec
            Start 155: cfg4Test.G4BoxTest
    155/155 Test #155: cfg4Test.G4BoxTest .........................   Passed    0.05 sec

    100% tests passed, 0 tests failed out of 155

    Total Test time (real) =  48.30 sec


