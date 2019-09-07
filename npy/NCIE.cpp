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


#include "NCIE.hpp"
#include "ciexyz.h"


float cie_X(float nm)
{
    float X = xFit_1931(nm);
    return X ;
}

float cie_Y(float nm)
{
    float Y = yFit_1931(nm);
    return Y ;
}

float cie_Z(float nm)
{
    float Z = zFit_1931(nm);
    return Z ;
}


float NCIE::X(float nm) { return xFit_1931(nm); }
float NCIE::Y(float nm) { return yFit_1931(nm); }
float NCIE::Z(float nm) { return zFit_1931(nm); }



