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
#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"


template <typename T> 
struct BRAP_API BLocSeqDigest
{
     // defaults correspond to the step-seqs, 
     // more strict reqes used for the track-seq
    BLocSeqDigest(bool skipdupe=true, bool requirematch=false, unsigned dump_loc_min=0);

    bool                                           _skipdupe ; 
    bool                                           _requirematch ;
    unsigned                                       _dump_loc_min ; 
 
    std::vector<std::string>                       _locs ; 
    std::map<std::string, unsigned>                _digest_count ; 
    std::map<std::string, T>                       _digest_marker ; 
    std::map<std::string, std::string>             _digest_locations ; 

    void add(const char* loc);
    void mark(T marker);

    void dumpDigests(const char* msg, bool locations) const ; 
    void dumpLocations(const std::vector<std::string>& digests) const ;

};

#include "BRAP_TAIL.hh"

