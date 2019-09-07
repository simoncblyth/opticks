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

#include "CFG4_BODY.hh"
#include "CMath.hh"

#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"

G4AffineTransform CMath::make_affineTransform(const G4Transform3D& T )
{
    G4ThreeVector colX(T.xx(), T.xy(), T.xz());
    G4ThreeVector colY(T.yx(), T.yy(), T.yz());
    G4ThreeVector colZ(T.zx(), T.zy(), T.zz());

    G4RotationMatrix rot(colX,colY,colZ) ;
    G4ThreeVector tlate(T.dx(), T.dy(), T.dz());

    return G4AffineTransform( rot, tlate) ; 
}

