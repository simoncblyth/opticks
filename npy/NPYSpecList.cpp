#include <iterator>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <sstream>

#include "NPYSpecList.hpp"
#include "NPYSpec.hpp"


NPYSpecList::NPYSpecList()
{
}

void NPYSpecList::add( unsigned idx, const NPYSpec* spec )
{   
    m_idx.push_back(idx); 
    m_spec.push_back(spec); 
}

unsigned NPYSpecList::getNumSpec() const
{
    assert( m_idx.size() == m_spec.size()); 
    return m_idx.size(); 
}

const NPYSpec* NPYSpecList::getByIdx(unsigned idx) const 
{
    typedef std::vector<unsigned>::const_iterator IT ; 
    IT it = std::find( m_idx.begin() , m_idx.end(), idx );
    return it == m_idx.end() ? NULL : m_spec[ std::distance( m_idx.begin(), it ) ] ; 
}
        
std::string NPYSpecList::description() const 
{
    std::stringstream ss ; 
    unsigned num_spec = getNumSpec(); 
    for(unsigned i=0 ; i < num_spec ; i++)
    {
        unsigned idx = m_idx[i] ; 
        const NPYSpec* spec = m_spec[i] ; 
        ss 
           << std::setw(5) << idx 
           << " : "
           << spec->description()
           << std::endl 
          ; 
    }
    return ss.str(); 
}
 
