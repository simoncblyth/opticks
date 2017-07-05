
// primitives
#include "NSphere.hpp"
#include "NZSphere.hpp"
#include "NBox.hpp"
#include "NSlab.hpp"
#include "NPlane.hpp"
#include "NCylinder.hpp"
#include "NDisc.hpp"
#include "NCone.hpp"
#include "NConvexPolyhedron.hpp"



/*




===================   =============  ================  =================
primitive              parametric     dec_z1/inc_z2 
===================   =============  ================  ================= 
nbox                    Y              N   
ncone                   Y              Y                 kludged parametric endcap/body join
nconvexpolyhedron       N(*)           N                 hmm : defined by planes ? minimally provide single point for each plane
ncylinder               Y              Y                 kludged para 
ndisc                   Y              Y                 kludged para + need flexibility wrt uv steps for different surfs : ie just 1+1 in z for disc
nnode                   -              -   
nplane                  -              -   
nslab                   -              -   
nsphere                 Y              N   
nzsphere                Y              Y   
===================   =============  ================  ================= 



Primitives without z1/z2 parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* nbox 

  * CSG_BOX3 : unplaced, symmetric xyz sizes  
  * TODO: CSG_BOX4 with z1/z2 controls ?

* nsphere
* nconvexpolyhedron


Test Primitives not used in actual geometry 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* nslab
* nplane


*/




