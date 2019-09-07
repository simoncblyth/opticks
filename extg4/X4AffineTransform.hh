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

#pragma once

#include "X4_API_EXPORT.hh"

#include <string>
#include <glm/fwd.hpp>
#include "G4AffineTransform.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"

struct X4_API X4AffineTransform
{  
    static G4AffineTransform FromTransform(const G4Transform3D& T );
    static X4AffineTransform FromGLM( const glm::mat4& trs );

    X4AffineTransform( const G4AffineTransform&  t ); 

    G4RotationMatrix getRotation() const ;
    G4RotationMatrix getRotation_0() const ;
    std::string getRotationCode(const char* identifier) const ;

    G4ThreeVector getTranslation() const ;
    std::string getTranslationCode(const char* identifier) const ;

    bool isIdentityRotation() const ; 
    bool isIdentityTranslation() const ; 
    bool isIdentityTransform() const ; 
 

    const G4AffineTransform& tr ;    
    const G4RotationMatrix  rot ; 
    const G4ThreeVector     tla ;  

};
