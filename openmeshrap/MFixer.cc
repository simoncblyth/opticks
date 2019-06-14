#include <algorithm>
#include <sstream>

#include "GGeo.hh"
#include "GMeshLib.hh"
#include "GMesh.hh"

#include "MFixer.hh"
#include "MTool.hh"

#include "PLOG.hh"


MFixer::MFixer(GGeo* ggeo) : 
    m_ggeo(ggeo), 
    m_meshlib(ggeo->getMeshLib()),
    m_tool(NULL),
    m_verbose(false)
{
    init();
}

void MFixer::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}

void MFixer::init()
{
    m_tool = new MTool ; 
}

void MFixer::fixMesh()
{
    unsigned int nso = m_ggeo->getNumVolumes();
    unsigned int nme = m_ggeo->getNumMeshes();

    LOG(info) << "MFixer::fixMesh" 
              << " NumVolumes " << nso  
              << " NumMeshes " << nme 
              ; 

    typedef std::map<unsigned int, unsigned int> MUU ; 
    typedef MUU::const_iterator MUUI ; 

    typedef std::vector<unsigned int> VU ; 
    typedef std::map<unsigned int, VU > MUVU ; 

    
    MUU& mesh_usage = m_meshlib->getMeshUsage();
    MUVU& mesh_nodes = m_meshlib->getMeshNodes();

    for(MUUI it=mesh_usage.begin() ; it != mesh_usage.end() ; it++)
    {    
        unsigned int meshIndex = it->first ; 
        unsigned int nodeCount = it->second ; 

        VU& nodes = mesh_nodes[meshIndex] ;
        assert(nodes.size() == nodeCount );

        std::stringstream nss ; 
        for(unsigned int i=0 ; i < std::min( nodeCount, 5u ) ; i++) nss << nodes[i] << "," ;


        const GMesh* mesh = m_meshlib->getMeshWithIndex(meshIndex);
        gfloat4 ce = mesh->getCenterExtent(0);

        const char* shortName = mesh->getShortName();

        bool join = m_ggeo->shouldMeshJoin(mesh);

        unsigned int nv = mesh->getNumVertices() ; 
        unsigned int nf = mesh->getNumFaces() ; 
        unsigned int tc = m_tool->countMeshComponents(mesh); // topological components

        assert( tc >= 1 );  // should be 1, some meshes have topological issues however

        std::string& out = m_tool->getOut();
        std::string& err =  m_tool->getErr();
        unsigned int noise = m_tool->getNoise();

        const char* highlight = join ? "**" : "  " ; 

        bool dump = noise > 0 || tc > 1 || join || m_verbose ;
        //bool dump = true ; 

        if(dump)
            printf("  %4d (v%5d f%5d )%s(t%5d oe%5u) : x%10.3f : n%6d : n*v%7d : %40s : %s \n", meshIndex, nv, nf, highlight, tc, noise, ce.w, nodeCount, nodeCount*nv, shortName, nss.str().c_str() );

        if(noise > 0)
        {
            if(out.size() > 0 ) LOG(debug) << "out " << out ;  
            if(err.size() > 0 ) LOG(debug) << "err " << err ;  
        }

   }    


    for(MUUI it=mesh_usage.begin() ; it != mesh_usage.end() ; it++)
    {
        unsigned int meshIndex = it->first ; 
        const GMesh* mesh = m_meshlib->getMeshWithIndex(meshIndex);
        bool join = m_ggeo->shouldMeshJoin(mesh);
        if(join)
        {
             mesh->Summary("MFixer::meshFix joined mesh");
        }
    }
}



