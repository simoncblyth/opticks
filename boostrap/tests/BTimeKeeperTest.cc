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

#include <climits>

#include "BTimeKeeper.hh"

#include "BTimes.hh"
#include "BTimesTable.hh"


int main()
{
    BTimeKeeper tk ; 
    tk.start();

    for(int i=0 ; i < INT_MAX/100 ; i++) if(i%1000000 == 0 ) printf(".");
    printf("\n");
 
    tk("after loop");

    tk.stop();


    BTimesTable* tt = tk.makeTable();
    tt->save("$TMP");

    tt->dump();


    return 0 ; 
}
