#pragma once

#include <string>
#include <vector>

struct sdebug
{
    static constexpr const char* PREFIX = "    int " ; 
    static constexpr const char* SUFFIX = " ;" ; 
    static constexpr const char* LABELS = R"LITERAL(
    int addGenstep ;
    int beginPhoton ;
    int rjoinPhoton ;
    int pointPhoton ;
    int finalPhoton ;
    int d12match_fail ;
    int rjoinPhotonCheck ;
    int rjoinPhotonCheck_flag_AB ;
    int rjoinPhotonCheck_flag_MI ;
    int rjoinPhotonCheck_flag_xor ;
    int rjoinPhotonCheck_flagmask_AB ;
    int rjoinPhotonCheck_flagmask_MI ;
    int rjoinPhotonCheck_flagmask_or ;
    int rjoinSeqCheck ;
    int rjoinSeqCheck_flag_AB ;
    int rjoinSeqCheck_flag_MI ;
    int rjoinSeqCheck_flag_xor ;
)LITERAL";

    int addGenstep ;
    int beginPhoton ;
    int rjoinPhoton ;
    int pointPhoton ;
    int finalPhoton ;
    int d12match_fail ;
    int rjoinPhotonCheck ;
    int rjoinPhotonCheck_flag_AB ;
    int rjoinPhotonCheck_flag_MI ;
    int rjoinPhotonCheck_flag_xor ;
    int rjoinPhotonCheck_flagmask_AB ;
    int rjoinPhotonCheck_flagmask_MI ;
    int rjoinPhotonCheck_flagmask_or ;
    int rjoinSeqCheck ;
    int rjoinSeqCheck_flag_AB ;
    int rjoinSeqCheck_flag_MI ;
    int rjoinSeqCheck_flag_xor ;

    void zero(); 
    std::string desc() const ; 
};

inline void sdebug::zero(){ *this = {} ; }

#include <cassert>
#include <string>
#include <sstream>
#include <iomanip>

inline std::string sdebug::desc() const 
{
    std::vector<std::string> vars ; 
    sstr::PrefixSuffixParse(vars, PREFIX, SUFFIX, LABELS ); 
    bool expected_size = sizeof(sdebug) == sizeof(int)*vars.size();
    assert( expected_size ); 
    //const int* first = &addGenstep ;
    const int* first = (int*)this ;
    std::stringstream ss ; 
    ss << "sdebug::desc" << std::endl ; 
    for(int i=0 ; i < int(vars.size()) ; i++) ss << std::setw(40) << vars[i] << " : " << std::setw(10) << *(first+i) << std::endl; 
    std::string s = ss.str(); 
    return s ; 
}

