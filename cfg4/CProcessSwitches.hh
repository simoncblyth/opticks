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

#define USE_CUSTOM_CERENKOV
#define USE_CUSTOM_SCINTILLATION
#define USE_CUSTOM_BOUNDARY


//#define USE_DEBUG_TRANSPORTATION
//#define USE_POWER_THIRD_RAYLEIGH




#ifdef USE_CUSTOM_CERENKOV
class DsG4Cerenkov ; 
//class Cerenkov;
#else
class G4Cerenkov ;
#endif

#ifdef USE_CUSTOM_SCINTILLATION
class DsG4Scintillation ; 
//class Scintillation;
#else
class G4Scintillation ;
#endif

#ifdef USE_CUSTOM_BOUNDARY
class DsG4OpBoundaryProcess ; 
#else
class G4OpBoundaryProcess ; 
#endif


#ifdef USE_POWER_THIRD_RAYLEIGH
class DsG4OpRayleigh ; 
#else
class OpRayleigh ; 
#endif

#ifdef USE_DEBUG_TRANSPORTATION
class DebugG4Transportation ; 
#else
class G4Transportation ; 
#endif





