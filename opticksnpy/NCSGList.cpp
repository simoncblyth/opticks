#include <cstring>

#include "NCSG.hpp"
#include "NCSGList.hpp"

NCSGList::NCSGList(const char* csgpath, unsigned verbosity)
   :
   m_csgpath(strdup(csgpath)),
   m_verbosity(verbosity)
{
    int rc = NCSG::Deserialize( csgpath, m_trees, m_verbosity );
    assert( rc == 0 );
}

NCSG* NCSGList::getTree(unsigned index)
{
    return m_trees[index] ;
}

unsigned NCSGList::getNumTrees()
{
    return m_trees.size();
}

