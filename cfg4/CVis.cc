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


#include "G4Version.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "CVis.hh"


G4VisAttributes* CVis::MakeInvisible()
{
#if ( G4VERSION_NUMBER >= 1074 )
    return new G4VisAttributes(false) ;  // hjw
#else
    return new G4VisAttributes(G4VisAttributes::Invisible) ;
#endif
}

G4VisAttributes* CVis::MakeAtt(float r, float g, float b, bool wire)
{
     // g4-;g4-cls G4VisAttributes

    G4VisAttributes* att = new G4VisAttributes(G4Colour(r,g,b));
    //att->SetVisibility(true);
    if(wire) att->SetForceWireframe(true);

    //World_log->SetVisAttributes (G4VisAttributes::Invisible);

    return att ; 
}

