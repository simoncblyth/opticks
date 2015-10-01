
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>

#include <glm/glm.hpp>
//#include "GLMPrint.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// npy-
#include "stringutil.hpp"
#include "NPY.hpp"
#include "NCache.hpp"

//
#include "MWrap.hh"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;


template <typename MeshT>
inline void saveMesh(MeshT* mesh, char* dir, const char* postfix)
{
    typedef typename MeshT::VertexHandle VH ; 
    typedef typename MeshT::FaceHandle FH ; 
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::Point P ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 

    NCache cache(dir); 

    unsigned int nface = mesh->n_faces(); 
    unsigned int nvert = mesh->n_vertices(); 

    NPY<float>* vnpy = NPY<float>::make( nvert, 0, 3 ); 
    vnpy->zero(); 
    float* vertices = vnpy->getValues();

    NPY<float>* nnpy = NPY<float>::make( nvert, 0, 3 ); 
    nnpy->zero(); 
    float* normals = nnpy->getValues();

    NPY<float>* cnpy = NPY<float>::make( nvert, 0, 3 ); 
    cnpy->zero(); 
    float* colors = cnpy->getValues();


    NPY<int>*   inpy = NPY<int>::make( nface*3, 0, 1 ); 
    inpy->zero(); 
    int* indices = inpy->getValues();


    for(unsigned int i=0 ; i < nvert ; i++)
    {
        VH v = mesh->vertex_handle(i) ;   
        P p = mesh->point(v);
        P n = mesh->normal(v);

        vertices[i*3+0] = p[0] ;
        vertices[i*3+1] = p[1] ;
        vertices[i*3+2] = p[2] ;

        normals[i*3+0] = n[0] ;
        normals[i*3+1] = n[1] ;
        normals[i*3+2] = n[2] ;

        colors[i*3+0] = 0.5f ;
        colors[i*3+1] = 0.5f ;
        colors[i*3+2] = 0.5f ;

    }

    for(unsigned int i=0 ; i < nface ; i++)
    {
        FH fh = mesh->face_handle(i) ;   
        unsigned int j(0) ;   
        for(FVI fv=mesh->cfv_iter(fh) ; fv ; fv++ )
        {
            indices[i*3+j] = fv->idx(); 
            j++ ;
        } 
        assert(j == 3);
    }

    vnpy->setVerbose(true);
    nnpy->setVerbose(true);
    cnpy->setVerbose(true);
    inpy->setVerbose(true);

    vnpy->save( cache.path("GMergedMesh/0/vertices%s.npy", postfix).c_str() ); 
    nnpy->save( cache.path("GMergedMesh/0/normals%s.npy", postfix).c_str() ); 
    cnpy->save( cache.path("GMergedMesh/0/colors%s.npy", postfix).c_str() ); 
    inpy->save( cache.path("GMergedMesh/0/indices%s.npy", postfix).c_str() ); 

} 


int main()
{
    char* dir = getenv("JDPATH") ;

    MyMesh* src = new MyMesh ;
    MWrap<MyMesh> wsrc(src);

    bool dedupe = true ; 
    wsrc.loadFromMergedMesh(dir, 0, dedupe);

    src->request_face_normals();
    src->update_normals();

    int ncomp = wsrc.labelConnectedComponentVertices("component"); 
    printf("ncomp: %d \n", ncomp);

    wsrc.dump("wsrc", 0);

    if(ncomp != 2) return 1 ; 

    typedef MyMesh::VertexHandle VH ; 
    typedef std::map<VH,VH> VHM ;

    MWrap<MyMesh> wa(new MyMesh);
    MWrap<MyMesh> wb(new MyMesh);

    VHM s2c_0 ;  
    wsrc.partialCopyTo(wa.getMesh(), "component", 0, s2c_0);

    VHM s2c_1 ;  
    wsrc.partialCopyTo(wb.getMesh(), "component", 1, s2c_1);

    wa.dump("wa",0);
    wb.dump("wb",0);

    wa.write("/tmp/comp%d.off", 0 );
    wb.write("/tmp/comp%d.off", 1 );

    wa.calcFaceCentroids("centroid"); 
    wb.calcFaceCentroids("centroid"); 

    // xyz delta maximum and w: minimal dot product of normals, -0.999 means very nearly back-to-back
    glm::vec4 delta(10.f, 10.f, 10.f, -0.999 ); 

    MWrap<MyMesh>::labelSpatialPairs( wa.getMesh(), wb.getMesh(), delta, "centroid", "paired");

    wa.deleteFaces("paired");
    wb.deleteFaces("paired");

    wa.collectBoundaryLoop();
    wb.collectBoundaryLoop();

    VHM a2b = MWrap<MyMesh>::findBoundaryVertexMap(&wa, &wb );  

    MWrap<MyMesh> wdst(new MyMesh);

    wdst.createWithWeldedBoundary( &wa, &wb, a2b );

    saveMesh<MyMesh>( wdst.getMesh(), dir, "_v0");

    return 0;
}


