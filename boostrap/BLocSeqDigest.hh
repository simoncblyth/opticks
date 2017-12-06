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

