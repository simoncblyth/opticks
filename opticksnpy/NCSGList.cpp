#include <cstring>

#include "BBnd.hh"
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
    m_verbosity(verbosity),
    m_bndspec(NULL),
    m_universe(NULL),
    m_container_bbox()  
{
    init();
}

void NCSGList::init()
{
    init_bbox(m_container_bbox) ;
}

NCSG* NCSGList::getUniverse() const 
{
    return m_universe ; 
}

std::vector<NCSG*>& NCSGList::getTrees()
{
    return m_trees ; 
}

std::string NCSGList::getTreeDir(unsigned idx) const 
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

    m_bndspec = new NTxt(txtpath.c_str());
    m_bndspec->read();
    m_bndspec->dump("NCSGList::load");    

    unsigned nbnd = m_bndspec->getNumLines();

    LOG(info) << "NCSGList::load"
              << " VERBOSITY " << m_verbosity 
              << " basedir " << m_csgpath 
              << " txtpath " << txtpath 
              << " nbnd " << nbnd 
              ;


    // order is reversed so that a tree with the "container" meta data tag at tree slot 0
    // is handled last, so container_bb will then have been adjusted to hold all the others...
    // allowing the auto-bbox setting of the container
    //
    // this order flipping feels kludgy, 
    // but because of the export of the resultant bbox it aint easy to fix
    //


    for(unsigned j=0 ; j < nbnd ; j++)
    {
        unsigned idx = nbnd - 1 - j ;     // idx 0 is handled last 
    
        const char* boundary = m_bndspec->getLine(idx);

        NCSG* tree = loadTree(idx, boundary);

        nbbox bba = tree->bbox_analytic();  

       // for non-container trees updates m_container_bbox, for the container trees adopts the bbox 
        if(!tree->isContainer())
        {
            m_container_bbox.include(bba);
        } 
        else if(tree->isContainer())
        {
            float scale = tree->getContainerScale(); // hmm should be prop of the list not the tree ? 
            float delta = 0.f ; 
            tree->adjustToFit(m_container_bbox, scale, delta );
        }
      
        tree->export_(); // from CSG nnode tree back into *same* in memory buffer, with bbox added   

        LOG(debug) << "NCSGList::load [" << idx << "] " << tree->desc() ; 

        m_trees.push_back(tree);  
    }

    // back into original source order with outer first eg [outer, container, sphere]  
    std::reverse( m_trees.begin(), m_trees.end() );


    m_universe = createUniverse(1., 1.);
}

        

NCSG* NCSGList::createUniverse(float scale, float delta) const 
{
    const char* bnd0 = m_bndspec->getLine(0);
    const char* ubnd = BBnd::DuplicateOuterMaterial( bnd0 ); 

    LOG(info) << "NCSGList::createUniverse"
              << " bnd0 " << bnd0 
              << " ubnd " << ubnd
              << " scale " << scale
              << " delta " << delta
              ;
 
    NCSG* universe = loadTree(0, ubnd ) ;    // cheat clone 

    assert( !universe->isContainer() );

    universe->adjustToFit( m_container_bbox, scale, delta ); 

    return universe ; 
}


NCSG* NCSGList::loadTree(unsigned idx, const char* boundary) const 
{
    std::string treedir = getTreeDir(idx);

    NCSG* tree = new NCSG(treedir.c_str());

    tree->setIndex(idx);
    tree->setVerbosity( m_verbosity );
    tree->setBoundary( boundary );

    tree->load();    // m_nodes, the user input serialization buffer (no bbox from user input python)
    tree->import();  // input m_nodes buffer into CSG nnode tree 

    return tree ; 
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




void NCSGList::dumpDesc(const char* msg) const 
{
    LOG(info) << msg ; 

    unsigned numTrees = getNumTrees() ;

    std::cout << "NCSGList::dumpDesc"
              << " csgpath " << m_csgpath
              << " verbosity " << m_verbosity 
              << " numTrees " << numTrees
              << std::endl 
              ;

    for(unsigned i=0 ; i < numTrees ; i++)
    {
         NCSG* tree = getTree(i);
         std::cout 
             << " tree " << std::setw(2) << i << " "
             << tree->desc() 
             << std::endl ;
    }
}

void NCSGList::dumpMeta(const char* msg) const 
{
    LOG(info) << msg ; 

    unsigned numTrees = getNumTrees() ;

    std::cout << "NCSGList::dumpMeta"
              << " csgpath " << m_csgpath
              << " verbosity " << m_verbosity 
              << " numTrees " << numTrees
              << std::endl 
              ;

    for(unsigned i=0 ; i < numTrees ; i++)
    {
         NCSG* tree = getTree(i);
         std::cout 
             << " tree " << std::setw(2) << i << " "
             << tree->meta() 
             << std::endl ;
    }
}


void NCSGList::dump(const char* msg) const 
{
    dumpDesc(msg);
    dumpMeta(msg);
}

void NCSGList::dumpUniverse(const char* msg) const 
{
    LOG(info) << msg
              << " csgpath " << m_csgpath
              ;

    NCSG* tree = getUniverse();

    std::cout 
         << " meta " 
         << std::endl 
         << tree->meta() 
         << std::endl 
         << " desc " 
         << std::endl 
         << tree->desc() 
         << std::endl 
         ;

}






