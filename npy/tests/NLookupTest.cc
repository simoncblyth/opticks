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

#include "NLookup.hpp"

// canonical use in GGeo::setupLookup

int main(int, char**)
{
    NLookup::mockup("$TMP", "mockA.json", "mockB.json");

    NLookup lookup;
    lookup.loadA("$TMP", "mockA.json");
    lookup.loadB("$TMP", "mockB.json");
    lookup.crossReference();
    lookup.dump("NLookupTest");



/* 
#include "G4StepNPY.hpp"
    const char* det = "dayabay" ; 
    G4StepNPY cs(NPY<float>::load("cerenkov", "1", det));
    cs.setLookup(&lookup);
    cs.applyLookup(0, 2); // materialIndex  (1st quad, 3rd number)
    cs.dump("cs.dump");
    cs.dumpLines("");

*/

    return 0 ;
}



