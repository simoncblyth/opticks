old-style-analytic-raytrace-noshow
=====================================


Getting the triangulated raytrace ? Why ?

Two reasons:

1. had a bug in arguments : now FIXED

2. OPTICKS_RESOURCE_LAYOUT envvar is not passed thru to the launched process :
   not exactly true the : script "#!/bin/bash -l"  uses -l so the login
   shell trumps envvars passed like above 

::

    OPTICKS_RESOURCE_LAYOUT=103 op --gltf 3 
    OPTICKS_RESOURCE_LAYOUT=103 op --gltf 3  --dumpenv
    
    op --gltf 3  --dumpenv
        ## need to change it in .bash_profile

    OPTICKS_RESOURCE_LAYOUT=103 OTracerTest --gltf 3
         ## no script, so works directly 






