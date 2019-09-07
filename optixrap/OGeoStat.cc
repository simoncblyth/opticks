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

#include <sstream>
#include <iostream>
#include <iomanip>

#include "OGeoStat.hh"


OGeoStat::OGeoStat( unsigned mmIndex_, unsigned numPrim_, unsigned numPart_, unsigned numTran_, unsigned numPlan_ )
       :
       mmIndex(mmIndex_),
       numPrim(numPrim_),
       numPart(numPart_),
       numTran(numTran_),
       numPlan(numPlan_)
{
}

std::string OGeoStat::desc()
{
    std::stringstream ss ; 
    ss << " mmIndex " << std::setw(3) << mmIndex 
       << " numPrim " << std::setw(5) << numPrim 
       << " numPart " << std::setw(5) << numPart
       << " numTran(triples) " << std::setw(5) << numTran
       << " numPlan " << std::setw(5) << numPlan
       ;
    return ss.str(); 
}


