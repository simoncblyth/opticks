#include <cstring>

#include "NCSG.hpp"
#include "NCSGList.hpp"

#include "PLOG.hh"

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


void NCSGList::dump(const char* msg)
{
    LOG(info) << msg ; 

    unsigned numTrees = getNumTrees() ;

    std::cout << "NCSGList::dump"
              << " csgpath " << m_csgpath
              << " verbosity " << m_verbosity 
              << " numTrees " << numTrees
              << std::endl 
              ;


    for(unsigned i=0 ; i < numTrees ; i++)
    {
         NCSG* tree = getTree(i);
         std::cout << tree->desc() << std::endl ;
    }

    

}


