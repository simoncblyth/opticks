opticks-full-faster
====================


Having to keep building multiple config is tedious - can opticks-full be made faster ?
----------------------------------------------------------------------------------------

After changing config/PREFIX to use fresh folders - have to opticks-full because of externals
----------------------------------------------------------------------------------------------


The externals are currently needed for every config despite being the same, 
because the externals directory is within the prefix.

::

    [lo] A[blyth@localhost opticks]$ glm-vi
    [lo] A[blyth@localhost opticks]$ glm-dir
    /home/blyth/tmp/local/opticks_Debug_g411/externals/glm/glm-0.9.9.5
    [lo] A[blyth@localhost opticks]$ t glm-dir
    glm-dir () 
    { 
        echo $(opticks-prefix)/externals/glm/$(glm-name)
    }


This layout is assumed in too many placed to change - but could perhaps
use a symbolic link ?




