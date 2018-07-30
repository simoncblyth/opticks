#pragma once

#include <vector>
#include <string>

#include "NPY_API_EXPORT.hh"
class NPYSpec ; 

class NPY_API NPYSpecList 
{
    public:
        NPYSpecList(); 
        void add( unsigned idx, const NPYSpec* spec ); 

        unsigned          getNumSpec() const ; 
        const NPYSpec*    getByIdx(unsigned idx) const ;
        std::string       description() const ; 
    private:
        std::vector<unsigned> m_idx ; 
        std::vector<const NPYSpec*> m_spec ; 

};  
