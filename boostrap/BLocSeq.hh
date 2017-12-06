#pragma once


#include <string>
#include <map>
#include <vector>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

template <typename T> 
class BRAP_API BLocSeq
{
    public:
        BLocSeq(bool skipdupe);

        void add(const char* loc, int record_id, int step_id); 
        void poststep();
        void mark(T marker); 

    public:
        void dump(const char* msg) const ; 
    private:
        void dumpDigests(const char* msg, bool locations) const ; 
        void dumpLocations(const std::vector<std::string>& digests) const ;
        void dumpCounts(const char* msg) const ; 

    private:
        bool                          m_skipdupe ; 
        unsigned                      m_global_flat_count ; 
        unsigned                      m_step_flat_count ; 
        unsigned                      m_count_mismatch; 

        std::map<unsigned, unsigned>                   m_record_count ; 
        std::vector<std::string>                       m_location_vec ; 
        std::map<std::string, unsigned>                m_digest_count ; 
        std::map<std::string, T>                       m_digest_marker ; 
        std::map<std::string, std::string>             m_digest_locations ; 


};

#include "BRAP_TAIL.hh"

