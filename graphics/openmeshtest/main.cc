
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


#include "md5digest.hpp"
#include "NPY.hpp"


#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;


struct Cache 
{
   Cache(char* dir) : cache(dir) {} ;

   std::string path(const char* relative)
   {
       fs::path cpath(cache/relative); 
       return cpath.string();
   } 

   fs::path cache ; 
};



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
inline void populateMesh(MeshT* mesh, char* dir)
{
   // Loads and de-duplicates vertices from *dir*
   // and populates *mesh* with the vertices and faces.
   //
   // Developed with single/few mesh caches in mind like --jdyb --kdyb 
   //

   Cache cache(dir); 

   NPY<float>* vertices_ = NPY<float>::load( cache.path("GMergedMesh/0/vertices.npy").c_str() ); 
   NPY<int>* faces_      = NPY<int>::load( cache.path("GMergedMesh/0/indices.npy").c_str() ); 

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

   typedef typename MeshT::VertexHandle VH ; 
   typedef typename MeshT::Point P ; 

   VH* vh = new VH[dd_vertices.num] ;

   float* vdata = dd_vertices.data ;
   assert(dd_vertices.elem == 3); 
   for(unsigned int i=0 ; i < dd_vertices.num ; i++)
   {
       vh[i] = mesh->add_vertex(P(*(vdata), *(vdata+1), *(vdata+2)));
       vdata += 3 ; 
   } 

   std::vector<VH>  face_vhandles;

   assert(dd_faces.elem == 3); 
   int* fdata = dd_faces.data ;
   for(unsigned int i=0 ; i < dd_faces.num ; i++)
   {
       face_vhandles.clear();

       int v0 = *(fdata + 0) ; 
       int v1 = *(fdata + 1) ; 
       int v2 = *(fdata + 2) ; 
       fdata += 3 ; 

       //printf( "f %4d : v %3d %3d %3d \n", i, v0, v1, v2 ); 
       face_vhandles.push_back(vh[v0]);
       face_vhandles.push_back(vh[v1]);
       face_vhandles.push_back(vh[v2]);
       mesh->add_face(face_vhandles);
   }

   delete[] vh ; 
}



template <typename MeshT>
inline int labelConnectedComponents(MeshT* mesh)
{
    OpenMesh::VPropHandleT<int> component ; 
    assert(true == mesh->get_property_handle(component, "component"));

    typedef typename MeshT::VertexHandle VH ;
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::VertexVertexIter VVI ;

    for( VI vi=mesh->vertices_begin() ; vi != mesh->vertices_end(); ++vi ) 
         mesh->property(component, *vi) = -1 ;

    VI seed = mesh->vertices_begin();
    VI end  = mesh->vertices_end();

    int componentIndex = -1 ; 
    while(true)
    {
        // starting from current look for unvisited "-1" vertices
        bool found_seed(false) ; 
        for(VI vi=seed ; vi != end ; vi++)
        {
            if(mesh->property(component, *vi) == -1) 
            {
                componentIndex += 1 ; 
                mesh->property(component, *vi) = componentIndex ;  
                seed = vi ; 
                found_seed = true ; 
                break ;  
            }
        }

        if(!found_seed) break ;  // no more unvisited vertices

        std::vector<VH> vstack ;
        vstack.push_back(*seed);

        // stack based recursion spreading the componentIndex to all connected vertices

        while(vstack.size() > 0)
        {
            VH current = vstack.back();
            vstack.pop_back();
            for (VVI vvi=mesh->vv_iter( current ); vvi ; ++vvi)
            {
                if(mesh->property(component, *vvi) == -1) 
                {
                    mesh->property(component, *vvi) = componentIndex ; 
                    vstack.push_back( *vvi );
                }
            }
        }
    }
    return componentIndex + 1 ; 
}


struct BBox {
   glm::vec3 min ; 
   glm::vec3 max ; 
};


template <typename MeshT>
inline void findBounds(MeshT* mesh, BBox& bb )
{
    bb.min = glm::vec3(FLT_MAX);
    bb.max = glm::vec3(-FLT_MAX);

    typedef typename MeshT::Point P ; 
    typedef typename MeshT::ConstVertexIter VI ; 
    for( VI vi=mesh->vertices_begin() ; vi != mesh->vertices_end(); ++vi )
    {
        P p = mesh->point(*vi) ; 

        bb.min.x = std::min( bb.min.x, p[0]);  
        bb.min.y = std::min( bb.min.y, p[1]);  
        bb.min.z = std::min( bb.min.z, p[2]);

        bb.max.x = std::max( bb.max.x, p[0]);  
        bb.max.y = std::max( bb.max.y, p[1]);  
        bb.max.z = std::max( bb.max.z, p[2]);
    }
}



template <typename MeshT>
inline void calcFCentroid(MeshT* mesh)
{
    typedef typename MeshT::FaceIter FI ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 
    typedef typename MeshT::Point P ; 

    OpenMesh::FPropHandleT<bool> copied;
    mesh->add_property(copied); 

    OpenMesh::FPropHandleT<P> centroid;
    mesh->add_property(centroid, "centroid");

    for( FI fi=mesh->faces_begin() ; fi != mesh->faces_end(); ++fi ) 
    {
        P cog ;
        mesh->calc_face_centroid( *fi, cog);
        mesh->property(centroid,*fi) = cog ; 
    }
}



template <typename MeshT>
inline void matchFCentroids(MeshT* a, MeshT* b, glm::vec4 delta)  
{
    calcFCentroid<MeshT>(a); 
    calcFCentroid<MeshT>(b); 

    typedef typename MeshT::Point P ; 
    typedef typename MeshT::FaceIter FI ; 

    OpenMesh::FPropHandleT<bool> a_cleaved ;
    a->add_property(a_cleaved, "cleaved"); 

    OpenMesh::FPropHandleT<bool> b_cleaved ;
    b->add_property(b_cleaved, "cleaved"); 

    for( FI af=a->faces_begin() ; af != a->faces_end(); ++af ) a->property(a_cleaved, *af) = false ; 
    for( FI bf=b->faces_begin() ; bf != b->faces_end(); ++bf ) b->property(b_cleaved, *bf) = false ; 


    OpenMesh::FPropHandleT<P> a_centroid ;
    assert(a->get_property_handle(a_centroid, "centroid"));

    OpenMesh::FPropHandleT<P> b_centroid ;
    assert(b->get_property_handle(b_centroid, "centroid"));

    // very inefficent approach, calculating all pairings 
    // but geometry surgery is a once only endeavor

    unsigned int npair(0);

    for( FI af=a->faces_begin() ; af != a->faces_end(); ++af ) 
    {
        int fa = af->idx(); 
        P ap = a->property(a_centroid, *af) ;
        P an = a->normal(*af);

        for( FI bf=b->faces_begin() ; bf != b->faces_end(); ++bf ) 
        { 
            int fb = bf->idx(); 
            P bp = b->property(b_centroid, *bf) ;
            P dp = bp - ap ; 
            P bn = b->normal(*bf);

            float adotb = OpenMesh::dot(an,bn); 

            bool close = fabs(dp[0]) < delta.x && fabs(dp[1]) < delta.y && fabs(dp[2]) < delta.z ;
            bool backtoback = adotb < delta.w ;  

            if(close && backtoback) 
            {
                 std::cout 
                       << std::setw(3) << npair 
                       << " (" << std::setw(3) << fa
                       << "," << std::setw(3) << fb 
                       << ")" 
                       << " an " << std::setprecision(3) << std::fixed << std::setw(20)  << an 
                       << " bn " << std::setprecision(3) << std::fixed << std::setw(20)  << bn 
                       << " a.b " << std::setprecision(3) << std::fixed << std::setw(10) << adotb
                       << " dp " << std::setprecision(3) << std::fixed << std::setw(20)  << dp
                       << std::endl ;  
                 npair++ ; 

                 // mark the cleaved faces
                 a->property(a_cleaved, *af ) = true ; 
                 b->property(b_cleaved, *bf ) = true ; 

            }
        }
    }
}

template <typename MeshT>
inline void deleteCleavedFaces(MeshT* mesh)  
{
    typedef typename MeshT::FaceIter FI ; 

    OpenMesh::FPropHandleT<bool> cleaved ;
    assert(mesh->get_property_handle(cleaved, "cleaved")); 

    LOG(info) << "deleteCleavedFaces " ; 
    for( FI f=mesh->faces_begin() ; f != mesh->faces_end(); ++f )
    {
        if(mesh->property(cleaved, *f)) 
        {
            std::cout << f->idx() << " " ; 
            bool delete_isolated_vertices = true ; 
            mesh->delete_face( *f, delete_isolated_vertices );
        }
    }
    std::cout << std::endl ; 
}
 

template <typename MeshT>
inline void dump(MeshT* mesh, const char* msg, unsigned int detail)
{
    unsigned int nface = std::distance( mesh->faces_begin(), mesh->faces_end() );
    unsigned int nvert = std::distance( mesh->vertices_begin(), mesh->vertices_end() );
    unsigned int nedge = std::distance( mesh->edges_begin(), mesh->edges_end() );

    unsigned int n_face = mesh->n_faces(); 
    unsigned int n_vert = mesh->n_vertices(); 
    unsigned int n_edge = mesh->n_edges();

    assert( nface == n_face );
    assert( nvert == n_vert );
    assert( nedge == n_edge );


    LOG(info) << msg  
              << " nface " << nface 
              << " nvert " << nvert 
              << " nedge " << nedge 
              << " V - E + F = " << nvert - nedge + nface 
              << " (should be 2 for Euler Polyhedra) "   
              ; 

    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::FaceIter FI ; 
    typedef typename MeshT::VertexFaceIter VFI ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 
    typedef typename MeshT::Point P ; 

    OpenMesh::FPropHandleT<P> centroid ;
    bool fcentroid = mesh->get_property_handle(centroid, "centroid");


    if(detail > 0)
    {
        for( FI fi=mesh->faces_begin() ; fi != mesh->faces_end(); ++fi ) 
        {
            int f_idx = fi->idx() ;  
            std::cout << " f " << std::setw(4) << *fi 
                      << " i " << std::setw(3) << f_idx 
                      << " v " << std::setw(3) << mesh->valence(*fi) 
                      << " : " 
                      
                      ; 

            if(fcentroid)
            {
                std::cout << " c "
                          << std::setprecision(3) << std::fixed << std::setw(20) 
                          << mesh->property(centroid,*fi)
                          << "  " 
                          ;
            } 

            // over points of the face 
            for(FVI fvi=mesh->cfv_iter(*fi) ; fvi ; fvi++) 
                 std::cout << std::setw(3) << *fvi << " " ;

            for(FVI fvi=mesh->cfv_iter(*fi) ; fvi ; fvi++) 
                 std::cout 
                           << std::setprecision(3) << std::fixed << std::setw(20) 
                           << mesh->point(*fvi) << " "
                           ;

             std::cout 
                  << " n " 
                  << std::setprecision(3) << std::fixed << std::setw(20) 
                  << mesh->normal(*fi)
                  << std::endl ;  

        }
    }

    if(detail > 0)
    {
        for( VI vi=mesh->vertices_begin() ; vi != mesh->vertices_end(); ++vi )
        {
             std::cout << " vi " << std::setw(3) << *vi << " # " << std::setw(3) << mesh->valence(*vi) << " : "  ;  
             // all faces around a vertex, fans are apparent
             for(VFI vfi=mesh->vf_iter(*vi)  ; vfi ; vfi++) 
                 std::cout << " " << std::setw(3) << *vfi ;   
             std::cout << std::endl ;  

             if(detail > 1)
             {
                 for(VFI vfi=mesh->vf_iter(*vi)  ; vfi ; vfi++) 
                 {
                     // over points of the face 
                    std::cout << "     "  ;  
                    for(FVI fvi=mesh->cfv_iter(*vfi) ; fvi ; fvi++) 
                         std::cout << std::setw(3) << *fvi << " " ;

                   for(FVI fvi=mesh->cfv_iter(*vfi) ; fvi ; fvi++) 
                          std::cout 
                           << std::setprecision(3) << std::fixed << std::setw(20) 
                           << mesh->point(*fvi) << " "
                           ;

                    std::cout 
                        << " n " 
                        << std::setprecision(3) << std::fixed << std::setw(20) 
                        << mesh->normal(*vfi)
                        << std::endl ;  
                 } 
             } 
        }
    }

    BBox bb ; 
    findBounds<MeshT>(mesh, bb);

    print( bb.max , "bb.max"); 
    print( bb.min , "bb.min"); 
    print( bb.max - bb.min , "bb.max - bb.min"); 
}


template <typename MeshT>
inline void populateComponentMesh(MeshT* comp, MeshT* mesh, int wanted )
{
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::FaceIter FI ; 
    typedef typename MeshT::VertexFaceIter VFI ; 
    typedef typename MeshT::VertexHandle VH ; 
    typedef typename MeshT::FaceHandle FH ; 
    typedef typename MeshT::Point P ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 

    OpenMesh::VPropHandleT<int> component ;
    assert(true == mesh->get_property_handle(component, "component"));

    OpenMesh::FPropHandleT<bool> copied;
    mesh->add_property(copied); 

    for( FI fi=mesh->faces_begin() ; fi != mesh->faces_end(); ++fi ) 
         mesh->property(copied, *fi) = false ;

    std::map<VH, VH> o2n ; 

    for( VI vi=mesh->vertices_begin() ; vi != mesh->vertices_end(); ++vi )
    { 
        if(mesh->property(component, *vi) != wanted ) continue ; 
        o2n[*vi] = comp->add_vertex(mesh->point(*vi)) ;
    }

    for( VI vi=mesh->vertices_begin() ; vi != mesh->vertices_end(); ++vi )
    {
        if(mesh->property(component, *vi) != wanted ) continue ; 

        for(VFI vfi=mesh->vf_iter(*vi) ; vfi ; vfi++) // all faces around a vertex, fans are apparent
        {
            if(mesh->property(copied, *vfi) == true) continue ;                 

            mesh->property(copied, *vfi) = true ; 

            // collect handles of the vertices of this fresh face 
            std::vector<VH>  fvh ;
            for(FVI fvi=mesh->cfv_iter(*vfi) ; fvi ; fvi++) fvh.push_back( o2n[*fvi] );

            comp->add_face(fvh);
        }       
    }

    comp->request_face_normals();
    comp->update_normals();

    // the request has to be called before a vertex/face/edge can be deleted. it grants access to the status attribute
    //  http://www.openmesh.org/Daily-Builds/Doc/a00058.html
    comp->request_face_status();
    comp->request_edge_status();
    comp->request_vertex_status();

}


template <typename MeshT>
inline void write(MeshT* mesh, const char* tmpl, unsigned int index)
{
  char path[128] ;
  snprintf( path, 128, tmpl, index );

  LOG(info) << "write " << path ; 
  try
  {
    if ( !OpenMesh::IO::write_mesh(*mesh, path) )
    {
      std::cerr << "Cannot write mesh to file" << std::endl;
    }
  }
  catch( std::exception& x )
  {
    std::cerr << x.what() << std::endl;
  }
}



int main()
{
    MyMesh* mesh = new MyMesh ;

    populateMesh<MyMesh>(mesh, getenv("JDPATH"));

    mesh->request_face_normals();
    mesh->update_normals();

    OpenMesh::VPropHandleT<int> component;
    mesh->add_property(component, "component"); 

    int ncomp = labelConnectedComponents<MyMesh>(mesh); 
    printf("ncomp: %d \n", ncomp);

    dump(mesh, "mesh", 0);

    MyMesh* comps = new MyMesh[ncomp] ;
    for(unsigned int i=0 ; i < ncomp ; i++)
    {
        populateComponentMesh<MyMesh>(comps + i, mesh, i);

        dump(comps + i, "comp", i == 0 ? 1 : 0 );
        write(comps + i, "/tmp/comp%d.off", i );
    }
 

    glm::vec4 delta(10.f, 10.f, 10.f, -0.999 ); // xyz delta maximum and w: minimal dot product of normals, -0.999 means very nearly back-to-back
    if(ncomp == 2)
    {
        matchFCentroids<MyMesh>( comps+0, comps+1, delta);

        deleteCleavedFaces<MyMesh>( comps+0 );
        deleteCleavedFaces<MyMesh>( comps+1 );
    }


    return 0;
}


