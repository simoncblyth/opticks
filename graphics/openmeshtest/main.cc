
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>

#include <glm/glm.hpp>
#include "GLMPrint.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// npy-
#include "md5digest.hpp"
#include "NPY.hpp"
#include "NCache.hpp"

#include "MWrap.hh"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;


std::string float3digest( float* data )
{
    MD5Digest dig ;
    dig.update( (char*)data, sizeof(float)*3 );
    return dig.finalize(); 
}


template <typename T>
struct Ary 
{
    // Ary owns the data ptr memory  
    Ary(T* data, unsigned int num , unsigned int elem) : data(data), num(num), elem(elem) {} ;

    ~Ary()
    { 
       //printf("Ary dtor\n");
       delete[] data ; 
    }

    T* data ; 
    unsigned int num  ; 
    unsigned int elem ; 
};


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


template <typename MeshT>
inline void loadMesh(MeshT* mesh, char* dir)
{
   // Loads and de-duplicates vertices from *dir*
   // and populates *mesh* with the vertices and faces.
   //
   // Developed with single/few mesh caches in mind like --jdyb --kdyb 
   //
   NCache cache(dir); 

   NPY<float>* vertices_ = NPY<float>::load( cache.path("GMergedMesh/0/vertices.npy").c_str() ); 
   NPY<int>* faces_      = NPY<int>::load( cache.path("GMergedMesh/0/indices.npy").c_str() ); 
   NPY<int>* nodes_      = NPY<int>::load( cache.path("GMergedMesh/0/nodes.npy").c_str() ); 

   Ary<float> vertices( vertices_->getValues(), vertices_->getShape(0), 3 );
   Ary<int>   faces(       faces_->getValues(), faces_->getShape(0)/3, 3 ); // at NPY level indices have shape (3n, 1) rather than (n,3)

   // de-duping vertices is mandatory  

   std::map<std::string, unsigned int> vtxmap ; 

   // vertex index translation, o2n: original into new de-duped and back n2o 
 
   int* o2n = new int[vertices.num] ;        
   int* n2o = new int[vertices.num] ;      

   unsigned int vidx = 0 ;   // new de-duped vertex index, into array to be created
   for(unsigned int i=0 ; i < vertices.num ; i++)
   {
       std::string dig = float3digest(vertices.data + 3*i);
       if(vtxmap.count(dig) == 0)  // unseen vertex based on digest identity
       {
           vtxmap[dig] = vidx ; 
           n2o[vidx] = i ; 
           vidx += 1 ; 
       }
      
       o2n[i] = vtxmap[dig]  ;

#ifdef DEBUG
       printf(" %4d : %4d : %10.3f %10.3f %10.3f : %s \n", i, o2n[i], 
                 *(vertices.data+3*i+0), *(vertices.data+3*i+1), *(vertices.data+3*i+2), dig.c_str() );
#endif
   }

   Ary<float> dd_vertices( new float[vidx*3],    vidx , 3 );
   Ary<int>   dd_faces(    new int[faces.num*3], faces.num , 3 );

   // copy old vertices to new leaving out the dupes ... 

   for(int n=0 ; n < vidx ; ++n )
   {
       int o = n2o[n] ;
       //printf(" n %4d n2o %4d \n", n, o );

       *(dd_vertices.data + n*3 + 0 ) = *(vertices.data + 3*o + 0) ;   
       *(dd_vertices.data + n*3 + 1 ) = *(vertices.data + 3*o + 1) ;   
       *(dd_vertices.data + n*3 + 2 ) = *(vertices.data + 3*o + 2) ;   
   }

   // map the vertex indices in the faces from old to new 

   for(unsigned int f=0 ; f < faces.num ; ++f )
   {
       int o0 = *(faces.data + 3*f + 0) ; 
       int o1 = *(faces.data + 3*f + 1) ; 
       int o2 = *(faces.data + 3*f + 2) ;

       *(dd_faces.data + f*3 + 0 ) = o2n[o0] ;
       *(dd_faces.data + f*3 + 1 ) = o2n[o1] ;
       *(dd_faces.data + f*3 + 2 ) = o2n[o2] ;
   }

   delete[] o2n ; 
   delete[] n2o ; 


   MWrap<MeshT> wmesh(mesh);
   wmesh.copyIn( dd_vertices.data, dd_vertices.num, dd_faces.data, dd_faces.num );
}




int main()
{
    char* dir = getenv("JDPATH") ;

    MyMesh* src = new MyMesh ;

    loadMesh<MyMesh>(src, dir);

    src->request_face_normals();
    src->update_normals();

    OpenMesh::VPropHandleT<int> component;
    src->add_property(component, "component"); 

    MWrap<MyMesh> wsrc(src);
    int ncomp = wsrc.labelConnectedComponents(); 
    printf("ncomp: %d \n", ncomp);

    wsrc.dump("src mesh", 1);

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


