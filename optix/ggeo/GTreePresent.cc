#include  "GTreePresent.hh"
#include "GGeo.hh"
#include "GMesh.hh"
#include "GSolid.hh"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GTreePresent::traverse()
{
    GSolid* top = m_ggeo->getSolid(m_top);
    traverse(top, 0, 0, 0, false);
}

void GTreePresent::traverse( GNode* node, unsigned int depth, unsigned int numSibling, unsigned int siblingIndex, bool elide )
{

    std::string indent(depth, ' ');
    int numChildren = node->getNumChildren() ;
    int nodeIndex   = node->getIndex() ; 
    unsigned int ridx = node->getRepeatIndex();   
    const char* name = node->getName() ; 

    //GSolid* solid = dynamic_cast<GSolid*>(node) ;
    //bool selected = solid->isSelected();

    std::stringstream ss ; 
    ss 
       << "  " << std::setw(7) << nodeIndex 
       << " [" << std::setw(3) << depth 
       << ":"  << std::setw(4) << siblingIndex 
       << "/"  << std::setw(4) << numSibling
       << "] " << std::setw(4) << numChildren   
       << " (" << std::setw(2) << ridx 
       << ") " << indent 
       << "  " << node->getName()
       << "  " << node->getMesh()->getName()
       << "  " << ( elide ? "..." : " " ) 
       ;

    m_flat.push_back(ss.str()); 
    if(elide) return ; 

    unsigned int hmax = m_sibling_max/2 ;

    if(depth < m_depth_max)
    {
       if( numChildren < 2*hmax )
       { 
           for(unsigned int i = 0; i < numChildren ; i++) 
               traverse(node->getChild(i), depth + 1, numChildren, i, false );
       }
       else
       {
           for(unsigned int i = 0; i < hmax ; i++) 
               traverse(node->getChild(i), depth + 1, numChildren, i, false);

           traverse(node->getChild(hmax), depth + 1, numChildren, hmax, true );

           for(unsigned int i = numChildren - hmax ; i < numChildren ; i++) 
               traverse(node->getChild(i), depth + 1, numChildren, i, false );
       }
    }
}



void GTreePresent::dump(const char* msg)
{
    LOG(info) << msg ; 
    std::copy(m_flat.begin(), m_flat.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
}

void GTreePresent::write(const char* dir)
{
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        if (fs::create_directory(cachedir))
        {
            printf("GTreePresent::write created directory %s \n", dir );
        }
    }

    if(fs::exists(cachedir) && fs::is_directory(cachedir))
    {

        fs::path txtpath(dir);
        txtpath /= "GTreePresent.txt" ; 
        const char* path = txtpath.string().c_str();
        LOG(info) << "GTreePresent::write " << path ;  

        { 
            std::ofstream fp(path, std::ios::out );
            std::copy(m_flat.begin(), m_flat.end(), std::ostream_iterator<std::string>(fp, "\n"));
        }

    }
    else
    {
        printf("GTreePresent::wrte directory %s DOES NOT EXIST \n", dir);
    }
}



