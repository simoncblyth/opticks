/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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
#include "NTorus.hpp"
#include "NHyperboloid.hpp"
#include "NCubic.hpp"
#include "NMultiUnion.hpp"
#include "NThetaCut.hpp"
#include "NPhiCut.hpp"


/*

Current primitives:

=======================  =================   ==================
 Type code                Python name         nnode sub-struct
=======================  =================   ==================
 CSG_SPHERE               sphere              nsphere 
 CSG_ZSPHERE              zsphere             nzsphere
 CSG_CYLINDER             cylinder            ncylinder  
 CSG_CONE                 cone                ncone
 CSG_DISC                 disc                ndisc
 CSG_BOX3                 box3                nbox
 CSG_BOX                  box                 nbox
 CSG_CONVEXPOLYHEDRON     convexpolyhedron    nconvexpolyhedron
 CSG_TRAPEZOID            trapezoid           nconvexpolyhedron
 CSG_SEGMENT              segment             nconvexpolyhedron
 CSG_TORUS                torus               ntorus
 CSG_HYPERBOLOID          hyperboloid         nhyperboloid
 CSG_CUBIC                cubic               ncubic
=======================  =================   ==================


Machinery primitives or not currently used:

=======================  =================  
 Type code                Python name         
=======================  =================  
 CSG_PLANE                plane
 CSG_SLAB                 slab
 CSG_UNDEFINED             
-----------------------  -----------------
 CSG_PRISM                prism
 CSG_PMT                  pmt
 CSG_ZLENS                zlens
 CSG_TUBS                 tubs
 CSG_MULTICONE            multicone
=======================  =================  








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




