
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

   std::string path(const char* tmpl, const char* incl)
   {
       char p[128];
       snprintf(p, 128, tmpl, incl);
       fs::path cpath(cache/p); 
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
inline void saveMesh(MeshT* mesh, char* dir, const char* postfix)
{
    typedef typename MeshT::VertexHandle VH ; 
    typedef typename MeshT::FaceHandle FH ; 
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::Point P ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 

    Cache cache(dir); 

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
   Cache cache(dir); 

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
#ifdef DEBUG
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
#endif
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

    unsigned int count(0);

    for( FI f=mesh->faces_begin() ; f != mesh->faces_end(); ++f )
    {
        if(mesh->property(cleaved, *f)) 
        {
           // std::cout << f->idx() << " " ; 
            bool delete_isolated_vertices = true ; 
            mesh->delete_face( *f, delete_isolated_vertices );
            count++ ; 
        }
    }
    //std::cout << std::endl ; 

    mesh->garbage_collection();
    LOG(info) << "deleteCleavedFaces " << count ; 
}


template <typename MeshT>
inline void collectBoundaryVertices(MeshT* mesh, std::vector<typename MeshT::VertexHandle>& vbnd)  
{
    typedef typename MeshT::FaceIter FI ; 
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::VertexHandle VH ;
    typedef typename MeshT::HalfedgeHandle HH ;
    typedef typename MeshT::Point P ; 


    VI v ; 
    VI v_end = mesh->vertices_end();

    for( v=mesh->vertices_begin(); v != v_end; ++v) if (mesh->is_boundary(*v)) break;

    if( v == v_end )
    {
        LOG(warning) << "collectBoundaryVertices : No boundary found\n";
        return;
    }
    // collect boundary loop
    HH hh0 = mesh->halfedge_handle(*v) ;
    HH hh = hh0 ;
    do 
    {
        vbnd.push_back(mesh->to_vertex_handle(hh));
        hh = mesh->next_halfedge_handle(hh);
    }
    while (hh != hh0);

#ifdef DEBUG
    std::cout << "collectBoundaryVertices collected " << vbnd.size() << std::endl ; 
    for(unsigned int i=1 ; i < vbnd.size() ; i++)
    {
        int a_idx = vbnd[i-1].idx();
        int b_idx = vbnd[i].idx();
        P a = mesh->point(vbnd[i-1]) ;
        P b = mesh->point(vbnd[i]) ;  
        P d = b - a ; 
        std::cout 
                << std::setw(3) << i
                << " (" << std::setw(3) << a_idx
                << "->" << std::setw(3) << b_idx 
                << ")" 
                << " a " << std::setprecision(3) << std::fixed << std::setw(20)  << a
                << " b " << std::setprecision(3) << std::fixed << std::setw(20)  << b 
                << " d " << std::setprecision(3) << std::fixed << std::setw(20)  << d
                << std::endl ;  
    }
#endif

}


template <typename MeshT>
inline void alignBoundaries(MeshT* a, MeshT* b, std::map<typename MeshT::VertexHandle, typename MeshT::VertexHandle>& a2b, std::vector<typename MeshT::VertexHandle>& abnd)
{
    LOG(info) << "alignBoundaries" ; 

    typedef typename MeshT::VertexHandle VH ;
    typedef typename MeshT::Point P ; 
    typedef typename std::vector<VH>::iterator VHI ; 

    std::vector<VH> bbnd ; 

    collectBoundaryVertices<MeshT>( a, abnd );
    collectBoundaryVertices<MeshT>( b, bbnd );

    for(VHI av=abnd.begin() ; av != abnd.end() ; av++ )
    {
        P ap = a->point(*av);
        int ai = av->idx();

        float bmin(FLT_MAX);
        VH bclosest = bbnd[0] ; 

        for(VHI bv=bbnd.begin() ; bv != bbnd.end() ; bv++ )
        {
            P bp = b->point(*bv);
            int bi = bv->idx();

            P dp = bp - ap ; 
            float dpn = dp.norm();
          
            if( dpn < bmin )
            {
                bmin = dpn ; 
                bclosest = *bv ; 
            }   
#ifdef DEBUG
            std::cout 
                << " (" << std::setw(3) << ai
                << "->" << std::setw(3) << bi 
                << ")" 
                << " ap " << std::setprecision(3) << std::fixed << std::setw(20)  << ap
                << " bp " << std::setprecision(3) << std::fixed << std::setw(20)  << bp 
                << " dp " << std::setprecision(3) << std::fixed << std::setw(20)  << dp
                << " dpn " << std::setprecision(3) << std::fixed << std::setw(10)  << dpn
                << std::endl ;  
#endif
        } 
        
        P bpc = b->point(bclosest);
        P dpc = bpc - ap ; 
        int bic = bclosest.idx();

        a2b[*av] = bclosest ; 

        std::cout 
                << " (" << std::setw(3) << ai
                << "->" << std::setw(3) << bic 
                << ")" 
                << " ap " << std::setprecision(3) << std::fixed << std::setw(20)  << ap
                << " bpc " << std::setprecision(3) << std::fixed << std::setw(20)  << bpc 
                << " dpc " << std::setprecision(3) << std::fixed << std::setw(20)  << dpc
                << " dpcn " << std::setprecision(3) << std::fixed << std::setw(10)  << dpc.norm()
                << std::endl ;  

    }
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
inline void copyMesh(MeshT* dst, MeshT* src, std::map<typename MeshT::VertexHandle, typename MeshT::VertexHandle>& src2dst )
{
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::FaceIter FI ; 
    typedef typename MeshT::VertexFaceIter VFI ; 
    typedef typename MeshT::VertexHandle VH ; 
    //typedef typename MeshT::FaceHandle FH ; 
    //typedef typename MeshT::Point P ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 

    OpenMesh::FPropHandleT<bool> copied;
    src->add_property(copied); 

    for( FI f=src->faces_begin() ; f != src->faces_end(); ++f ) 
         src->property(copied, *f) = false ;

    std::map<VH, VH> o2n ; 

    enum { VERT, FACE, NUM_PASS } ;

    for(unsigned int pass=VERT ; pass < NUM_PASS ; pass++) 
    {
        for( VI v=src->vertices_begin() ; v != src->vertices_end(); ++v )
        { 
            switch(pass)
            {
                case VERT:
                     src2dst[*v] = dst->add_vertex(src->point(*v)) ;
                     break ; 
                case FACE:
                     for(VFI vf=src->vf_iter(*v) ; vf ; vf++) 
                     {
                         if(src->property(copied, *vf) == true) continue ;                 
                         src->property(copied, *vf) = true ; 

                         // collect handles of the vertices of this fresh face, and use map to get corresponding dst handles 
                         std::vector<VH>  fvh ;
                         for(FVI fv=src->cfv_iter(*vf) ; fv ; fv++) fvh.push_back( src2dst[*fv] );
                         dst->add_face(fvh);
                     }  
                     break ; 
                default:
                     assert(0);
                     break ; 
            }
       }
    }

    dst->request_face_normals();
    dst->request_vertex_normals();
    dst->update_normals();
}



template <typename MeshT>
inline void weldBoundaries(MeshT* c, MeshT* a, MeshT* b) 
{ 

    typedef typename MeshT::VertexHandle VH ;
    typedef typename std::vector<VH> VHV ;
    typedef typename std::map<VH,VH> VHM ;
    typedef typename VHM::iterator VHMI ;
    typedef typename VHV::iterator VHVI ;

    VHV abnd ; // ordered vh around the A border  
    VHM a2b ;  // map from A vh across to B vh
    alignBoundaries<MeshT>(a, b, a2b, abnd );

    LOG(info) << "weldBoundaries " << a2b.size() ; 

 /* 
   Imagine the difficult to draw diagonals 

        +---+---+       +---+---+
    A   |   |   |       |   |   |  C 
        |   |   |       |   |   |
        +---+---+       +---+---+
    gap . \ . / .       |   |   |     
        +---+---+       +---+---+
        |   |   |       |   |   |
    B   |   |   |       |   |   |
        +---+---+       +---+---+


    Assume the gap is small enough to not need any new vertices, 
    just some new faces to fill out the flange.

    1 direct edge across to nearest, 
    1 diagonal across to next along other meshes boundary  
    2 tri-faces for each pair-of-pairs of verts 

    

  */

    VHM a2c ;
    copyMesh<MeshT>(c, a, a2c );

    VHM b2c ;
    copyMesh<MeshT>(c, b, b2c );


    LOG(info) << "weldBoundaries map ordering " ;


    for(VHMI j=a2b.begin() ; j != a2b.end() ; j++)
    {
        VH av = j->first ;  
        VH bv = j->second ;  

        VH avc = a2c[av] ;  // convert into mesh c handles
        VH bvc = b2c[bv] ; 

        std::cout 
             << "(" << av.idx() << "->" << bv.idx() << ")" 
             << "(" << avc.idx() << "->" << bvc.idx() << ")" 
             << std::endl ; 
    }


    LOG(info) << "weldBoundaries around boundary ordering " ;



    for(VHVI v=abnd.begin() ; v != abnd.end() ; v++)
    {
        VH av = *v ; 
        VH bv = a2b[av] ;

        // convert handles from a and b lingo into c lingo
        VH avc = a2c[av] ;  
        VH bvc = b2c[bv] ; 

        std::cout 
             << "(" << av.idx() << "->" << bv.idx() << ")" 
             << "(" << avc.idx() << "->" << bvc.idx() << ")" 
             << std::endl ; 
    }


    /*
         
           |           |           |
       A   |           |           |
        --a0----------a1----------a2
           .  f0   .   .           
           .  .   f1   .
        --b0----------b1----
      B    |           |
           |           |


    */

    bool flip = false ; 

    for(unsigned int i=1 ; i < abnd.size() ; i++)
    {
         VH a0 = abnd[i-1] ; 
         VH a1 = abnd[i] ; 

         VH b0 = a2b[a0] ; 
         VH b1 = a2b[a1] ; 

         VH a0c = a2c[a0] ; 
         VH a1c = a2c[a1]   ; 

         VH b0c = b2c[b0] ; 
         VH b1c = b2c[b1]   ; 
         
        std::cout 
             << " a0 " << std::setw(3) << a0.idx() 
             << " a1 " << std::setw(3) << a1.idx() 
             << " b0 " << std::setw(3) << b0.idx() 
             << " b1 " << std::setw(3) << b1.idx() 
             << " a0c " << std::setw(3) << a0c.idx() 
             << " a1c " << std::setw(3) << a1c.idx() 
             << " b0c " << std::setw(3) << b0c.idx() 
             << " b1c " << std::setw(3) << b1c.idx() 
             << std::endl ; 

         // how to use/check consistent winding order ?

         if(flip)
         {
            // causes lots of complex edge errors
             c->add_face( a1c, a0c, b0c );
             c->add_face( b0c, b1c, a1c );
         }
         else
         {
             c->add_face( a0c, a1c, b0c );
             c->add_face( b1c, b0c, a1c );
         } 
    }

    // boundary goes around a loop, does the tail need special casing ?
    // hmm maybe there is a missing face between N-1 and 0 ?
    //
    c->request_face_normals();
    c->update_normals();

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

    for( FI f=mesh->faces_begin() ; f != mesh->faces_end(); ++f ) 
         mesh->property(copied, *f) = false ;

    std::map<VH, VH> o2n ; 

    enum { VERT, FACE, NUM_PASS } ;

    for(unsigned int pass=VERT ; pass < NUM_PASS ; pass++) 
    {
        for( VI v=mesh->vertices_begin() ; v != mesh->vertices_end(); ++v )
        { 
            if(mesh->property(component, *v) != wanted ) continue ; 
            switch(pass)
            {
                case VERT:
                     o2n[*v] = comp->add_vertex(mesh->point(*v)) ;
                     break ; 
                case FACE:
                     for(VFI vf=mesh->vf_iter(*v) ; vf ; vf++) 
                     {
                         if(mesh->property(copied, *vf) == true) continue ;                 
                         mesh->property(copied, *vf) = true ; 

                         // collect handles of the vertices of this fresh face 
                         std::vector<VH>  fvh ;
                         for(FVI fv=mesh->cfv_iter(*vf) ; fv ; fv++) fvh.push_back( o2n[*fv] );
                         comp->add_face(fvh);
                     }  
                     break ; 
                default:
                     assert(0);
                     break ; 
            }
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
    char* dir = getenv("JDPATH") ;

    MyMesh* src = new MyMesh ;
    loadMesh<MyMesh>(src, dir);

    src->request_face_normals();
    src->update_normals();

    OpenMesh::VPropHandleT<int> component;
    src->add_property(component, "component"); 

    int ncomp = labelConnectedComponents<MyMesh>(src); 
    printf("ncomp: %d \n", ncomp);

#ifdef DEBUG
    dump(mesh, "mesh", 0);
#endif


    MyMesh* comps = new MyMesh[ncomp] ;
    for(unsigned int i=0 ; i < ncomp ; i++)
    {
        populateComponentMesh<MyMesh>(comps + i, src, i);

#ifdef DEBUG
        dump(comps + i, "comp", i == 0 ? 1 : 0 );
        write(comps + i, "/tmp/comp%d.off", i );
#endif
    }


    MyMesh* dst = new MyMesh ;
    if(ncomp == 2)
    {
        glm::vec4 delta(10.f, 10.f, 10.f, -0.999 ); // xyz delta maximum and w: minimal dot product of normals, -0.999 means very nearly back-to-back
        matchFCentroids<MyMesh>( comps+0, comps+1, delta);

        deleteCleavedFaces<MyMesh>( comps+0 );
        deleteCleavedFaces<MyMesh>( comps+1 );

        weldBoundaries<MyMesh>(dst, comps+0, comps+1);
    
        saveMesh<MyMesh>( dst, dir, "_v0");
    }


    return 0;
}


