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

#include "OPTICKS_LOG.hh"
#include "BLog.hh"
#include "BStr.hh"
#include "BTxt.hh"
#include "BFile.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    int pindex = argc > 1 ? BStr::atoi(argv[1]) : 1872 ; 

    const char* logpath = BStr::concat<int>("$TMP/ox_", pindex, ".log") ; 
    const char* txtpath = BStr::concat<int>("$TMP/ox_", pindex, ".txt") ; 

    BLog* a = BLog::Load(logpath); 
    const std::vector<double>&  av = a->getValues() ; 
    a->setSequence(&av) ; 
    a->dump("a"); 
    a->write(txtpath); 

    BLog* b = BLog::Load(txtpath); 
    b->dump("b"); 

    int RC = BLog::Compare(a, b ); 
    assert( RC == 0 ) ; 


    return 0 ; 
}
