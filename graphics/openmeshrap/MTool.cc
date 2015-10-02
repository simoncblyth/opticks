#include "MTool.hh"





#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// ggeo-
#include "GMesh.hh"

//
#include "MWrap.hh"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;



unsigned int MTool::countMeshComponents(GMesh* gm)
{
    MWrap<MyMesh> wsrc(new MyMesh);

    wsrc.load(gm);

    unsigned int ncomp = wsrc.labelConnectedComponentVertices("component"); 

    return ncomp ; 
}


GMesh* MTool::joinSplitUnion(GMesh* mesh, const char* config)
{
    LOG(info) << "MTool::joinSplitUnion " << mesh->getIndex() ; 
    return mesh ; 
}




