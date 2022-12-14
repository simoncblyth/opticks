flipping_compilation_switches_like_WITH_PMTSIM_WITH_PMTFASTSIM
=================================================================



To Flip OFF : delete the cmake dir and rebuild
-------------------------------------------------

::
 
    jfs
    om
    rm -rf /usr/local/opticks/lib/cmake/pmtfastsim   

    u4
    touch CMakeLists.txt
    om


To Flip ON : reinstall the optional pkg and build
----------------------------------------------------


::

    jfs
    om

    u4
    touch CMakeLists.txt
    om

     
* touching CMakeLists.txt ensures that the config search for external pkgs is done



CAUTION : regarding executables only built when an external pkg is found
---------------------------------------------------------------------------

When flipping OFF, the old version of the executable will continue to 
run : but will not be updated by any changes

* that is confusing : so its better for the executable to be always built no matter the switch
* the executable can exits with an error saying the needed external is not available 







