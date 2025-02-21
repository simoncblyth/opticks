zepeng-Rocky-Linux-9-gcc11p3-compilation
=========================================


Hello Simon,

I got both Opticks and a simulation example using it run with a container. Now,
I’m trying to set up Opticks on another cluster. I could set up CUDA toolkit
with module load on it so I no longer need a container.

I have two errors when running opticks-full, and the error message is as this::

    /home/zepengli/opticks/CSG/CSGTarget.cc: In member function ‘int CSGTarget::getFrame(sframe&, int) const’:
    /home/zepengli/opticks/CSG/CSGTarget.cc:81:5: error: invalid use of incomplete type ‘struct sframe’
       81 |     fr.set_inst(inst_idx);
          |     ^~
    In file included from /home/zepengli/opticks/CSG/CSGFoundry.h:19,
                     from /home/zepengli/opticks/CSG/CSGTarget.cc:9:
    /home/zepengli/opticks/install/include/SysRap/SGeo.hh:31:8: note: forward declaration of ‘struct sframe’
       31 | struct sframe ;
          |        ^~~~~~
    compilation terminated due to -fmax-errors=1.
    gmake[2]: *** [CMakeFiles/CSG.dir/build.make:195: CMakeFiles/CSG.dir/CSGTarget.cc.o] Error 1
    gmake[1]: *** [CMakeFiles/Makefile2:942: CMakeFiles/CSG.dir/all] Error 2

I made the following changes to solve them::

    --- a/CSG/CSGSimtrace.hh
    +++ b/CSG/CSGSimtrace.hh
    @@ -13,6 +13,7 @@ The heart of this is CSGQuery on CPU intersect functionality using the csg heade
     #include "plog/Severity.h"
     #include <vector>
     #include "sframe.h"
    +#include "SGeo.hh"

     struct CSGFoundry ;
     struct SEvt ;

    diff --git a/sysrap/SGeo.hh b/sysrap/SGeo.hh
    index 1ddd4b132..1adc0dea7 100644
    --- a/sysrap/SGeo.hh
    +++ b/sysrap/SGeo.hh
    @@ -27,6 +27,7 @@ CSGFoundry instance down to it cast down to this SGeo protocol base.
     #include "plog/Severity.h"
     #include <string>
     #include "SYSRAP_API_EXPORT.hh"
    +#include "sframe.h"

     struct sframe ;
     struct stree ;

This was not seen on the other cluster using an Ubuntu 22.04 container. It’s
probably caused by the environment. Here is the information on the cluster::

    Rocky Linux 9
    CUDA 12.2
    cmake 3.24.3
    gcc 11.3

Thanks,
Zepeng







