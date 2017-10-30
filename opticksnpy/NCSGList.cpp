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

NCSG* NCSGList::getTree(unsigned index) const 
{
    return m_trees[index] ;
}

unsigned NCSGList::getNumTrees() const 
{
    return m_trees.size();
}


NCSG* NCSGList::findEmitter() const 
{
    unsigned numTrees = getNumTrees() ;
    NCSG* emitter = NULL ; 
    for(unsigned i=0 ; i < numTrees ; i++)
    {
        NCSG* tree = getTree(i);
        if(tree->isEmit())
        {
           assert( emitter == NULL && "not expecting more than one emitter" );
           emitter = tree ;
        }
    }
    return emitter ; 
}



void NCSGList::dump(const char* msg) const 
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


