#include <cstring>

#include "BStr.hh"
#include "BFile.hh"

#include "NBBox.hpp"
#include "NTxt.hpp"
#include "NCSG.hpp"
#include "NCSGList.hpp"

#include "PLOG.hh"

const char* NCSGList::FILENAME = "csg.txt" ; 


bool NCSGList::ExistsDir(const char* dir)
{
    if(!dir) return false ; 
    if(!BFile::ExistsDir(dir)) return false ; 
    return true ; 
}


NCSGList* NCSGList::Load(const char* csgpath, int verbosity)
{
    if(!NCSGList::ExistsDir(csgpath))
    {
        LOG(warning) << "NCSGList::Load missing csgpath " << csgpath ; 
        return NULL ; 
    }
    NCSGList* ls = new NCSGList(csgpath, verbosity );
    ls->load();
    return ls ;
} 



NCSGList::NCSGList(const char* csgpath, int verbosity)
    :
    m_csgpath(strdup(csgpath)),
    m_verbosity(verbosity)
{
}


std::vector<NCSG*>& NCSGList::getTrees()
{
    return m_trees ; 
}

std::string NCSGList::getTreeDir(unsigned idx)
{
    return BFile::FormPath(m_csgpath, BStr::itoa(idx));  
}


void NCSGList::load()
{
    assert(m_trees.size() == 0);

    std::string txtpath = BFile::FormPath(m_csgpath, FILENAME) ;
    bool exists = BFile::ExistsFile(txtpath.c_str() ); 

    if(!exists) LOG(fatal) << "NCSGList::load"
                           << " file does not exist " 
                           << txtpath 
                           ;
    assert(exists); 

    NTxt bnd(txtpath.c_str());
    bnd.read();
    bnd.dump("NCSGList::load");    

    unsigned nbnd = bnd.getNumLines();

    LOG(info) << "NCSGList::load"
              << " VERBOSITY " << m_verbosity 
              << " basedir " << m_csgpath 
              << " txtpath " << txtpath 
              << " nbnd " << nbnd 
              ;

    nbbox container_bb = make_bbox() ; 

    // order is reversed so that a tree with the "container" meta data tag at tree slot 0
    // is handled last, so container_bb will then have been adjusted to hold all the others...
    // allowing the auto-bbox setting of the container

    for(unsigned j=0 ; j < nbnd ; j++)
    {
        unsigned i = nbnd - 1 - j ;    
        std::string treedir = BFile::FormPath(m_csgpath, BStr::itoa(i));  

        NCSG* tree = new NCSG(treedir.c_str());
        tree->setIndex(i);
        tree->setVerbosity( m_verbosity );
        tree->setBoundary( bnd.getLine(i) );

        tree->load();    // m_nodes, the user input serialization buffer (no bbox from user input python)
        tree->import();  // input m_nodes buffer into CSG nnode tree 
        tree->updateContainer(container_bb); // for non-container trees updates container_bbox, for the container trees adopts the bbox 
        tree->export_(); // from CSG nnode tree back into *same* in memory buffer, with bbox added   

        LOG(debug) << "NCSGList::load [" << i << "] " << tree->desc() ; 

        m_trees.push_back(tree);  
    }

    // back into original source order with outer first eg [outer, container, sphere]  
    std::reverse( m_trees.begin(), m_trees.end() );

}






NCSG* NCSGList::getTree(unsigned index) const 
{
    return m_trees[index] ;
}

unsigned NCSGList::getNumTrees() const 
{
    return m_trees.size();
}


int NCSGList::polygonize()
{
    unsigned numTrees = getNumTrees() ;
    assert(numTrees > 0);

    LOG(info) << "NCSGList::polygonize"
              << " csgpath " << m_csgpath
              << " verbosity " << m_verbosity 
              << " ntree " << numTrees
              ;

    int rc = 0 ; 
    for(unsigned i=0 ; i < numTrees ; i++)
    {
        NCSG* tree = getTree(i); 
        tree->setVerbosity(m_verbosity);
        tree->polygonize();
        if(tree->getTris() == NULL) rc++ ; 
    }     
    return rc ; 
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






