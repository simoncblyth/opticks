//  grab lower dependency pieces from openmeshrap MWrap.cc as needed


#include <limits>
#include <iostream>
#include <sstream>

#include "PLOG.hh"
#include "NGLM.hpp"
#include "Nuv.hpp"
#include "NNode.hpp"
#include "NOpenMesh.hpp"



template <typename T>
std::string NOpenMesh<T>::desc(const typename T::Point& pt)
{
    std::stringstream ss ; 
    ss 
        << " (" 
        << std::setw(15) << std::setprecision(3) << std::fixed << pt[0]
        << "," 
        << std::setw(15) << std::setprecision(3) << std::fixed << pt[1]
        << "," 
        << std::setw(15) << std::setprecision(3) << std::fixed << pt[2]
        << ") " 
        ;

    return ss.str();
}


template <typename T>
NOpenMesh<T>::NOpenMesh(const nnode* node, int level, int verbosity, float epsilon)
    :
    node(node), 
    level(level), 
    verbosity(verbosity), 
    epsilon(epsilon),
    nsubdiv(2),
    leftmesh(NULL),
    rightmesh(NULL) 
{
    init();
}

template <typename T>
const char* NOpenMesh<T>::F_INSIDE_OTHER = "f_inside_other" ; 

template <typename T>
const char* NOpenMesh<T>::V_SDF_OTHER = "v_sdf_other" ; 

template <typename T>
const char* NOpenMesh<T>::V_PARAMETRIC = "v_parametric" ; 



template <typename T>
void NOpenMesh<T>::init()
{
    OpenMesh::FPropHandleT<int> f_inside_other ;
    mesh.add_property(f_inside_other, F_INSIDE_OTHER);  

    OpenMesh::VPropHandleT<float> v_sdf_other ;
    mesh.add_property(v_sdf_other, V_SDF_OTHER);  

    OpenMesh::VPropHandleT<nuv> v_parametric ;
    mesh.add_property(v_parametric, V_PARAMETRIC);  


    // without the below get segv on trying to delete a face
    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_vertex_status();


    build_parametric();
}


template <typename T>
void NOpenMesh<T>::build_parametric()
{
/*

Aim is to combine leftmesh and rightmesh faces 
appropriately for the CSG operator of the combination. 

Faces that are entirely inside or outside the other 
can simply be copied or not copied as appropriate into the combined mesh.  

The complex part is how to stitch up the join between the two(? or more) open boundaries. 

union(A,B)
    copy faces of A entirely outside B and vv  (symmetric/commutative)

intersection(A,B)
    copy faces of A entirely inside B and vv (symmetric/commutative)

difference(A,B) = intersection(A,-B)      (asymmetric/not-commutative)
    copy faces of A entirely outside -B 
    copy faces of A entirely inside B 

    copy faces of A entirely outside B 
    copy faces of B entirely inside A 

*/

    bool combination = node->left && node->right ; 

    if(!combination)
    {
        build_parametric_primitive();  // adds unique vertices and faces to build out the parametric mesh  
    }
    else
    {
        assert(node->type == CSG_UNION || node->type == CSG_INTERSECTION || node->type == CSG_DIFFERENCE );
   
        typedef NOpenMesh<T> Mesh ; 
        leftmesh = new Mesh(node->left, level, verbosity, epsilon) ; 
        rightmesh = new Mesh(node->right, level, verbosity, epsilon) ; 

        LOG(info) << "build_parametric" 
                  << " leftmesh " << leftmesh
                  << " rightmesh " << rightmesh
                   ;


        leftmesh->mark_faces( node->right );
        LOG(info) << "[0] leftmesh inside node->right : " <<  leftmesh->desc_inside_other() ;  

        rightmesh->mark_faces( node->left );
        LOG(info) << "[0] rightmesh inside node->left : " <<  rightmesh->desc_inside_other() ;  

        /**

        0. sub-object mesh tris are assigned a facemask property from (0-7) (000b-111b) 
           indicating whether vertices have negative other sub-object SDF values  

        1. CSG sub-object mesh faces with mixed other sub-object sdf signs (aka border faces) 
           are subdivided (ie original tris are deleted and replaced with four new smaller ones)

        2. border subdiv is repeated in nsubdiv rounds, increasing mesh "resolution" around the border
        
        3. Only triangles with all 3 vertices inside or outside the other sub-object get copied
           into the composite mesh, this means there is no overlapping/intersection at this stage  

        4. remaining gap between sub-object meshes then needs to be zippered up 
           to close the composite mesh ???? how exactly 
     
        **/

        leftmesh->subdivide_border_faces( node->right, nsubdiv );
        LOG(info) << "[1] leftmesh inside node->right : " <<  leftmesh->desc_inside_other() ;  

        rightmesh->subdivide_border_faces( node->left, nsubdiv );
        LOG(info) << "[1] rightmesh inside node->left : " <<  rightmesh->desc_inside_other() ;  


        if(node->type == CSG_UNION)
        {
            copy_faces( leftmesh,  ALL_OUTSIDE_OTHER );
            copy_faces( rightmesh, ALL_OUTSIDE_OTHER );

            dump_border_faces( "border faces", 'L' );
        }
        else if(node->type == CSG_INTERSECTION)
        {
            copy_faces( leftmesh,  ALL_INSIDE_OTHER );
            copy_faces( rightmesh, ALL_INSIDE_OTHER );
        }
        else if(node->type == CSG_DIFFERENCE )
        {
            copy_faces( leftmesh,  ALL_OUTSIDE_OTHER );
            copy_faces( rightmesh, ALL_INSIDE_OTHER );
        }


        // hmm there should be two open border loops from left and right meshes... 
        // need to retain left/right info into the composite..

        typename T::HalfedgeHandle heh = find_boundary_halfedge() ;  
        std::cout << "find_boundary_halfedge: " << heh << std::endl ; 




    }
}

template <typename T>
bool NOpenMesh<T>::is_border_face(const int facemask)
{
    return !( facemask == ALL_OUTSIDE_OTHER || facemask == ALL_INSIDE_OTHER ) ; 
}


template <typename T>
void NOpenMesh<T>::subdivide_border_faces(const nnode* other, unsigned nsubdiv)
{
    OpenMesh::FPropHandleT<int> f_inside_other ;
    assert(mesh.get_property_handle(f_inside_other, F_INSIDE_OTHER));

    typedef typename T::FaceHandle          FH ; 
    typedef typename T::FaceIter            FI ; 

    std::vector<FH> border_faces ; 

    for(unsigned round=0 ; round < nsubdiv ; round++)
    {
        border_faces.clear();
        for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
        {
            FH fh = *f ;  
            int _f_inside_other = mesh.property(f_inside_other, fh) ; 
            if(!is_border_face(_f_inside_other)) continue ; 
            border_faces.push_back(fh);
        }

        LOG(info) << "subdivide_border_faces" 
                  << " nsubdiv " << nsubdiv
                  << " round " << round 
                  << " nbf: " << border_faces.size()
                  ;

        for(unsigned i=0 ; i < border_faces.size(); i++) subdivide_face(border_faces[i], other); 

        mesh.garbage_collection();  // NB this invalidates handles, so dont hold on to them
    }

}

template <typename T>
void NOpenMesh<T>::subdivide_face(typename T::FaceHandle fh, const nnode* other)
{
/*
Uniform subdivision of single triangle face

* add three new vertices at midpoints of original triangle edges 
* delete face, leaving triangular hole, retain vertices
* add 4 new faces


                     o[2]
                     +
                    / \
                   /   \
                  /     \
                 /  f[2] \
                /         \
         m[2]  /           \  m[1]
              +-------------+ 
             / \           / \
            /   \  f[3]   /   \
           /     \       /     \
          /  f[0] \     / f[1]  \
         /         \   /         \
        /           \ /           \
       +-------------+-------------+
      o[0]          m[0]           o[1]

*/

    //LOG(info) << "subdivide_face" << " fh " << fh ; 

    typedef typename T::FaceHandle          FH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::FaceHalfedgeIter    FHI ; 
    typedef typename T::Point               P ; 

    VH o[3] ; 
    VH m[3] ; 
    FH f[4] ; 

    unsigned n(0) ; 
    for(FHI fhe=mesh.cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
    {
        const HEH& heh = *fhe ; 

        const VH& fr_ = mesh.from_vertex_handle( heh );
        const VH& to_ = mesh.to_vertex_handle( heh );

        const P& fr = mesh.point( fr_ );
        const P& to = mesh.point( to_ );
        P mi = (fr + to)/2.f ; 

        bool added(false); 
        o[n] = fr_ ;
        m[n] = add_vertex_unique(mi, added, epsilon);  

        //assert(added == true);   
        // not always added, as edges (and midpoints) are shared 
         

        n++ ; 
    }
    assert( n == 3);


    bool delete_isolated_vertices = false ; 
    mesh.delete_face( fh, delete_isolated_vertices );

    f[0] = add_face_( o[0], m[0], m[2], verbosity ); 
    f[1] = add_face_( o[1], m[1], m[0], verbosity ); 
    f[2] = add_face_( m[1], o[2], m[2], verbosity ); 
    f[3] = add_face_( m[0], m[1], m[2], verbosity ); 

    mark_face(f[0], other);
    mark_face(f[1], other);
    mark_face(f[2], other);
    mark_face(f[3], other);

    //LOG(info) << "subdivide_face DONE" << " fh " << fh ; 



}
 


 
template <typename T>
void NOpenMesh<T>::copy_faces(const NOpenMesh<T>* other, int facemask)
{
    typedef typename T::FaceHandle          FH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::Point               P ; 

    OpenMesh::FPropHandleT<int> f_inside_other ;
    assert(other->mesh.get_property_handle(f_inside_other, F_INSIDE_OTHER));

    for( FI f=other->mesh.faces_begin() ; f != other->mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        int _f_inside_other = other->mesh.property(f_inside_other, fh) ; 

        if( _f_inside_other == facemask )
        {  
            VH nvh[3] ; 
            int fvert(0);
            for(FVI fv=other->mesh.cfv_iter(fh) ; fv.is_valid() ; fv++) 
            {
                const VH& vh = *fv ; 
                const P& pt = other->mesh.point(vh);
                bool added(false);        
                nvh[fvert++] = add_vertex_unique(P(pt[0], pt[1], pt[2]), added, epsilon);  
            } 
            assert( fvert == 3 );
            add_face_( nvh[0], nvh[1], nvh[2], verbosity ); 
        }
    }
    // Hmm how to retain pointers to open boundaries, to avoid a subsequent search.
}
 


template <typename T>
typename T::HalfedgeHandle NOpenMesh<T>::find_boundary_halfedge()
{
    typedef typename T::FaceIter            FI ; 
    typedef typename T::FaceHandle          FH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::Point               P ; 
    typedef typename T::FaceHalfedgeIter    FHI ; 
    typedef typename T::HalfedgeHandle      HEH ; 

    HEH heh ; 


/*
  for (v_it=mesh_.vertices_begin(); v_it!=v_end; ++v_it)
    if (mesh_.is_boundary(v_it.handle()))
      break;

  // boundary found ?
  if (v_it == v_end)
  {
    std::cerr << "No boundary found\n";
    return;
  }

  // collect boundary loop
  vh = v_it.handle();
  hh = mesh_.halfedge_handle(vh);
  do
  {
    loop.push_back(mesh_.to_vertex_handle(hh));
    hh = mesh_.next_halfedge_handle(hh);
  }
  while (hh != mesh_.halfedge_handle(vh));

*/



    for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        bool check_vertex = false ; 
        if(!mesh.is_boundary( fh, check_vertex )) continue ; 

        for(FHI fhe=mesh.cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
        {
            heh = *fhe ; 
            if(!mesh.is_boundary(heh)) continue ;  

            P pt[2];
            VH vh[2] ;  

            vh[0] = mesh.from_vertex_handle( heh );
            vh[1] = mesh.to_vertex_handle( heh );

            pt[0] = mesh.point( vh[0] );
            pt[1] = mesh.point( vh[1] );

            std::cout << "find_boundary"
                      << " fr " << vh[0] << desc(pt[0])
                      << " to " << vh[1] << desc(pt[1])
                      << std::endl 
                      ; 

        }
    } 

    return heh ; 
}
 






template <typename T>
void NOpenMesh<T>::dump_border_faces(const char* msg, char side)
{
    LOG(info) << msg  ; 

    typedef NOpenMesh<T> Mesh ; 

    Mesh* a_mesh = NULL  ;
    Mesh* b_mesh = NULL  ;
    assert( side == 'L' || side == 'R' );

    switch(side)
    {
       case 'L':{  
                   a_mesh = leftmesh ; 
                   b_mesh = rightmesh ; 
                }
                break ;

       case 'R':{  
                   a_mesh = rightmesh ; 
                   b_mesh = leftmesh ; 
                }
                break ;
    }

    const nnode* a_node = a_mesh->node ; 
    const nnode* b_node = b_mesh->node ; 

    std::function<float(float,float,float)> a_sdf = a_node->sdf() ; 
    std::function<float(float,float,float)> b_sdf = b_node->sdf() ; 


    typedef typename T::FaceHandle          FH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::ConstFaceEdgeIter   FEI ; 
    typedef typename T::ConstFaceHalfedgeIter   FHI ; 
    typedef typename T::Point               P ; 

    OpenMesh::FPropHandleT<int> f_inside_other ;
    assert(a_mesh->mesh.get_property_handle(f_inside_other, F_INSIDE_OTHER));

    OpenMesh::VPropHandleT<nuv> v_parametric;
    assert(a_mesh->mesh.get_property_handle(v_parametric, V_PARAMETRIC));


    for( FI f=a_mesh->mesh.faces_begin() ; f != a_mesh->mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        int _f_inside_other = a_mesh->mesh.property(f_inside_other, fh) ; 
        if(!is_border_face(_f_inside_other)) continue ; 
            
        std::cout << "facemask:" << _f_inside_other << std::endl ; 

        // a_mesh edges along which b_sdf changes sign can be bisected 
        // (can treat as unary functions as only one 
        //  parameter will vary along the parametric edge)
        //  does that stay true above the leaves ? need to arrange for it to stay true... 

      
        VH vh[2] ;  
        nuv uv[2] ; 
        glm::vec3 a_pos[2] ;
        P pt[2];
        float _a_sdf[2] ; 
        float _b_sdf[2] ; 
        bool pmatch[2] ; 


        for(FHI fhe=a_mesh->mesh.cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
        {
            const HEH& heh = *fhe ; 
            vh[0] = a_mesh->mesh.from_vertex_handle( heh );
            vh[1] = a_mesh->mesh.to_vertex_handle( heh );

            uv[0] = a_mesh->mesh.property(v_parametric, vh[0]) ; 
            uv[1] = a_mesh->mesh.property(v_parametric, vh[1]) ; 

            a_pos[0] = a_node->par_pos( uv[0] );
            a_pos[1] = a_node->par_pos( uv[1] );

            _a_sdf[0] = a_sdf( a_pos[0].x, a_pos[0].y, a_pos[0].z );
            _a_sdf[1] = a_sdf( a_pos[1].x, a_pos[1].y, a_pos[1].z );

            _b_sdf[0] = b_sdf( a_pos[0].x, a_pos[0].y, a_pos[0].z );
            _b_sdf[1] = b_sdf( a_pos[1].x, a_pos[1].y, a_pos[1].z );

            pt[0] = a_mesh->mesh.point(vh[0]);
            pt[1] = a_mesh->mesh.point(vh[1]);

            pmatch[0] = pt[0][0] == a_pos[0][0] && pt[0][1] == a_pos[0][1] && pt[0][2] == a_pos[0][2] ;
            pmatch[1] = pt[1][0] == a_pos[1][0] && pt[1][1] == a_pos[1][1] && pt[1][2] == a_pos[1][2] ;

            assert( _a_sdf[0] == 0.f );
            assert( _a_sdf[1] == 0.f );


            std::cout << " heh " << heh
                      << " vh " << vh[0] << " -> " << vh[1]
                      << " uv " << uv[0].desc() << " -> " << uv[1].desc() 
                      << " a_pos " << glm::to_string(a_pos[0]) << " -> " << glm::to_string(a_pos[1]) 
                      << " pmatch[0] " << pmatch[0]
                      << " pmatch[1] " << pmatch[1]
                      << " _b_sdf " 
                      << std::setw(15) << std::setprecision(3) << std::fixed << _b_sdf[0]
                      << " -> " 
                      << std::setw(15) << std::setprecision(3) << std::fixed << _b_sdf[1]
                      << std::endl ;  

        }
    }
}




template <typename T>
void NOpenMesh<T>::mark_faces(const nnode* other)
{
    typedef typename T::FaceHandle          FH ; 
    typedef typename T::FaceIter            FI ; 

    for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        mark_face( fh, other );
    }
}
 

template <typename T>
void NOpenMesh<T>::mark_face(typename T::FaceHandle fh, const nnode* other)
{

/*
facemask "f_inside_outside"
-----------------------------

bitmask of 3 bits for each face corresponding to 
inside/outside for the vertices of the face

0   (0) : all vertices outside other

1   (1) : 1st vertex inside other
2  (10) : 2nd vertex inside other
3  (11) : 1st and 2nd vertices inside other
4 (100) : 3rd vertex inside other
5 (101) : 1st and 3rd vertex inside other
6 (110) : 2nd and 3rd vertex inside other

7 (111) : all vertices inside other 

*/

    std::function<float(float,float,float)> sdf = other->sdf();

    OpenMesh::VPropHandleT<float> v_sdf_other ;
    assert(mesh.get_property_handle(v_sdf_other, V_SDF_OTHER));

    OpenMesh::FPropHandleT<int> f_inside_other ;
    assert(mesh.get_property_handle(f_inside_other, F_INSIDE_OTHER));


    typedef typename T::VertexHandle        VH ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::Point               P ; 

    int _f_inside_other = 0 ;  

    assert( mesh.valence(fh) == 3 );

    int fvert = 0 ; 

    for(FVI fv=mesh.cfv_iter(fh) ; fv.is_valid() ; fv++) 
    {
        const VH& vh = *fv ; 
    
        const P& pt = mesh.point(vh);

        float dist = sdf(pt[0], pt[1], pt[2]);

        mesh.property(v_sdf_other, vh ) = dist   ;

        bool inside_other = dist < 0.f ; 

        _f_inside_other |=  (!!inside_other << fvert++) ;   
    }
    assert( fvert == 3 );
    mesh.property(f_inside_other, fh) = _f_inside_other ; 


    if(f_inside_other_count.count(_f_inside_other) == 0)
    {
        f_inside_other_count[_f_inside_other] = 0 ; 
    }
    f_inside_other_count[_f_inside_other]++ ; 
}
 

template <typename T>
std::string NOpenMesh<T>::desc_inside_other()
{
    std::stringstream ss ; 
    typedef std::map<int,int> MII ; 

    for(MII::const_iterator it=f_inside_other_count.begin() ; it!=f_inside_other_count.end() ; it++)
    {
         ss << std::setw(3) << it->first << " : " << std::setw(6) << it->second << "|" ; 
    }

    return ss.str();
}
 


 



template <typename T>
int NOpenMesh<T>::write(const char* path)
{
    try
    {
      if ( !OpenMesh::IO::write_mesh(mesh, path) )
      {
        std::cerr << "Cannot write mesh to file " << path << std::endl;
        return 1;
      }
    }
    catch( std::exception& x )
    {
      std::cerr << x.what() << std::endl;
      return 1;
    }
    return 0 ; 
}

template <typename T>
void NOpenMesh<T>::dump(const char* msg)
{
    LOG(info) << msg << " " << brief() ; 
    dump_vertices();
    dump_faces();
}

template <typename T>
int NOpenMesh<T>::euler_characteristic()
{
    unsigned n_faces    = std::distance( mesh.faces_begin(),    mesh.faces_end() );
    unsigned n_vertices = std::distance( mesh.vertices_begin(), mesh.vertices_end() );
    unsigned n_edges    = std::distance( mesh.edges_begin(),    mesh.edges_end() );

    assert( n_faces    == mesh.n_faces() );
    assert( n_vertices == mesh.n_vertices() );
    assert( n_edges    == mesh.n_edges() );

    int euler = n_vertices - n_edges + n_faces  ;
    return euler ; 
}

template <typename T>
std::string NOpenMesh<T>::brief()
{
    std::stringstream ss ; 
    ss 
        << " V " << mesh.n_vertices()
        << " E " << mesh.n_edges()
        << " F " << mesh.n_faces()
        << " Euler [(V - E + F)] " << euler_characteristic() 
        ;
    return ss.str();
}


template <typename T>
bool NOpenMesh<T>::is_consistent_face_winding(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2)
{
    typedef typename T::HalfedgeHandle  HEH ; 
    typedef typename T::VertexHandle    VH ; 
   
    VH _vertex_handles[3] ; 
    _vertex_handles[0] = v0 ; 
    _vertex_handles[1] = v1 ; 
    _vertex_handles[2] = v2 ; 

    int i,ii,n(3) ; 

    struct WindingCheck  
    {
        HEH   halfedge_handle;
        bool is_new;
    };

    // checking based on PolyConnectivity::add_face

    std::vector<WindingCheck> edgeData_; 
    edgeData_.resize(n);

    for (i=0, ii=1; i<n; ++i, ++ii, ii%=n)
    {
        // Initialise edge attributes
        edgeData_[i].halfedge_handle = mesh.find_halfedge(_vertex_handles[i],
                                                          _vertex_handles[ii]);
        edgeData_[i].is_new = !edgeData_[i].halfedge_handle.is_valid();
  
        if (!edgeData_[i].is_new && !mesh.is_boundary(edgeData_[i].halfedge_handle))
        {
            std::cerr << "predicting... PolyMeshT::add_face: complex edge\n";
            return false ;
        }
    }
    return true ; 
}



template <typename T>
void NOpenMesh<T>::dump_vertices(const char* msg)
{
    LOG(info) << msg ; 

    typedef typename T::Point          P ; 
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::HalfedgeHandle HEH ; 
    typedef typename T::FaceHandle     FH ; 
    typedef typename T::Vertex         V ; 
    typedef typename T::VertexIter     VI ; 

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        int idx = vh.idx() ;
        assert( idx == std::distance( beg, vit ) ) ;

        const P& p = mesh.point(vh); 

        //const V& v = mesh.vertex(vh); // <-- purely private internal V has no public methods, so useless

        const HEH& heh = mesh.halfedge_handle(vh); 
        bool heh_valid = mesh.is_valid_handle(heh);


        std::cout 
             << " vh " << std::setw(5) << vh  
             << " p " 
             << "[" 
             << std::setw(15) << std::fixed << std::setprecision(4) << p[0] << ","
             << std::setw(15) << std::fixed << std::setprecision(4) << p[1] << ","
             << std::setw(15) << std::fixed << std::setprecision(4) << p[2] << ","
             << "]"
             << " heh " << std::setw(5) << heh  
             ;

        if(heh_valid)
        {
            const VH& tvh = mesh.to_vertex_handle(heh);
            const VH& fvh = mesh.from_vertex_handle(heh);
            const FH& fh  = mesh.face_handle(heh);
            bool bnd = mesh.is_boundary(heh);

            std::cout  
                << " fvh->tvh " 
                << std::setw(3) << fvh << "->" 
                << std::setw(3) << tvh   
                << " fh " << std::setw(5) << fh  
                << " bnd " << std::setw(5) << bnd 
                << std::endl ;
        }
        else
        {
             std::cout << std::endl ; 
        }

    }
}


template <typename T>
void NOpenMesh<T>::dump_faces(const char* msg )
{
    LOG(info) << msg << " nface " << mesh.n_faces() ; 

    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::Point               P ; 

    for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
    {
        int f_idx = f->idx() ;  
        std::cout << " f " << std::setw(4) << *f 
                  << " i " << std::setw(3) << f_idx 
                  << " v " << std::setw(3) << mesh.valence(*f) 
                  << " : " 
                  ; 

        // over points of the face 
        for(FVI fv=mesh.cfv_iter(*f) ; fv.is_valid() ; fv++) 
             std::cout << std::setw(3) << *fv << " " ;

        for(FVI fv=mesh.cfv_iter(*f) ; fv.is_valid() ; fv++) 
             std::cout 
                       << std::setprecision(3) << std::fixed << std::setw(20) 
                       << mesh.point(*fv) << " "
                       ;

        std::cout << std::endl ; 
    }
}
 

template <typename T>
void NOpenMesh<T>::add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3, int verbosity )
{
   /*
              3-------2
              |     . | 
              |   .   |
              | .     |
              0-------1  
   */

    add_face_(v0,v1,v2, verbosity);
    add_face_(v2,v3,v0, verbosity);
}
 
template <typename T>
void NOpenMesh<T>::build_cube()
{
    /*

                 3-----------2
                /|          /| 
               / |         / |
              0-----------1  |
              |  |        |  |
              |  |        |  |
              |  7--------|--6
              | /         | /
              |/          |/
              4-----------5
          

         z  y
         | /
         |/
         +---> x

    */

    typedef typename T::Point P ; 
    typename T::VertexHandle vh[8];

    vh[0] = mesh.add_vertex(P(-1, -1,  1));
    vh[1] = mesh.add_vertex(P( 1, -1,  1));
    vh[2] = mesh.add_vertex(P( 1,  1,  1));
    vh[3] = mesh.add_vertex(P(-1,  1,  1));
    vh[4] = mesh.add_vertex(P(-1, -1, -1));
    vh[5] = mesh.add_vertex(P( 1, -1, -1));
    vh[6] = mesh.add_vertex(P( 1,  1, -1));
    vh[7] = mesh.add_vertex(P(-1,  1, -1));

    add_face_(vh[0],vh[1],vh[2],vh[3]);
    add_face_(vh[7],vh[6],vh[5],vh[4]);
    add_face_(vh[1],vh[0],vh[4],vh[5]);
    add_face_(vh[2],vh[1],vh[5],vh[6]);
    add_face_(vh[3],vh[2],vh[6],vh[7]);
    add_face_(vh[0],vh[3],vh[7],vh[4]);
}





template <typename T>
typename T::VertexHandle NOpenMesh<T>::find_vertex_exact(typename T::Point pt)  
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::Point           P ; 
    typedef typename T::VertexIter     VI ; 

    VH result ;

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        const P& p = mesh.point(vh); 
        if( p == pt )
        {
            result = vh ; 
            break ;          
        }
    }
    return result ; 
}



template <typename T>
typename T::VertexHandle NOpenMesh<T>::find_vertex_closest(typename T::Point pt, float& distance )  
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::Point           P ; 
    typedef typename T::VertexIter     VI ; 

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    VH closest ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        const P& p = mesh.point(vh); 

        P d = p - pt ; 
        float dlen = d.length();

        if(dlen < distance )
        {
            closest = vh ;
            distance = dlen ;  
        }
    }
    return closest ; 
}


template <typename T>
typename T::VertexHandle NOpenMesh<T>::find_vertex_epsilon(typename T::Point pt, const float epsilon )  
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::Point           P ; 

    float distance = std::numeric_limits<float>::max() ;

    VH empty ; 
    VH closest = find_vertex_closest(pt, distance );
         
    return distance < epsilon ? closest : empty ;  
}


template <typename T>
typename T::VertexHandle NOpenMesh<T>::add_vertex_unique(typename T::Point pt, bool& added, const float epsilon )  
{
    typedef typename T::VertexHandle VH ;

    VH prior = find_vertex_epsilon(pt, epsilon ) ;

    bool valid = mesh.is_valid_handle(prior) ;

    added = valid ? false : true  ;  

    return valid ? prior : mesh.add_vertex(pt)   ; 
}


template <typename T>
typename T::FaceHandle NOpenMesh<T>::add_face_(typename T::VertexHandle v0, typename T::VertexHandle v1, typename T::VertexHandle v2, int verbosity )  
{
    typedef typename T::FaceHandle FH ; 

    assert( v0 != v1 );
    assert( v0 != v2 );
    assert( v1 != v2 );

    if(verbosity > 3)
    {
        std::cout << "add_face_"
                  << std::setw(5) << v0 
                  << std::setw(5) << v1 
                  << std::setw(5) << v2
                  << std::endl ;  

    }

    FH fh = mesh.add_face(v0,v1,v2);
    assert(mesh.is_valid_handle(fh));
    return fh ; 
}





template <typename T>
void NOpenMesh<T>::build_parametric_primitive()  
{
   /*
   Singularities like the poles of a latitude/longitude sphere parametrization 
   are handled by detecting the degenerate verts and adjusting the
   faces generated accordingly. 

   NB parameterizations should avoid leaning on the epsilon crutch, better 
   to provide exactly equal positions for poles and seams    

   */

    int nu = 1 << level ; 
    int nv = 1 << level ; 


    OpenMesh::VPropHandleT<nuv> v_parametric ;
    assert(mesh.get_property_handle(v_parametric, V_PARAMETRIC));


    int ns = node->par_nsurf() ;
    auto vid = [ns,nu,nv](int s, int u, int v) { return  s*(nu+1)*(nv+1) + v*(nu+1) + u ; };

    int num_vert = (nu+1)*(nv+1)*ns ; 

    if(verbosity > 0)
    LOG(info) << "NOpenMesh<T>::build_parametric"
              << " ns " << ns
              << " nu " << nu
              << " nv " << nv
              << " num_vert(raw) " << num_vert 
              << " epsilon " << epsilon
              ;


    typedef typename T::VertexHandle VH ;
    typedef typename T::Point P ; 

    VH* vh = new VH[num_vert] ;

    int umin = 0 ; int umax = nu ; 
    int vmin = 0 ; int vmax = nv ; 

    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = vmin; v <= vmax ; v++)
        {
            for (int u = umin; u <= umax ; u++) 
            {
                nuv uv = make_uv(s,u,v,nu,nv);

                glm::vec3 pos = node->par_pos(uv);

                bool added(false) ;

                int vidx = vid(s,u,v) ;

                vh[vidx] = add_vertex_unique(P(pos.x, pos.y, pos.z), added, epsilon) ; 

                if(added)
                {
                    mesh.property(v_parametric, vh[vidx] ) = uv   ;
                } 
         
            }
        }
    }

    



    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = vmin; v < vmax; v++){
        for (int u = umin; u < umax; u++) 
        {
            int i00 = vid(s,u    ,     v) ;
            int i10 = vid(s,u + 1,     v) ;
            int i11 = vid(s,u + 1, v + 1) ;
            int i01 = vid(s,u    , v + 1) ;

            VH v00 = vh[i00] ;
            VH v10 = vh[i10] ;
            VH v01 = vh[i01] ;
            VH v11 = vh[i11] ;


            if(verbosity > 2)
            std::cout 
                  << " s " << std::setw(3)  << s
                  << " v " << std::setw(3)  << v
                  << " u " << std::setw(3)  << u
                  << " v00 " << std::setw(3)  << v00
                  << " v10 " << std::setw(3)  << v10
                  << " v01 " << std::setw(3)  << v01
                  << " v11 " << std::setw(3)  << v11
                  << std::endl 
                  ;



         /*


            v
            ^
            4---5---6---7---8
            | / | \ | / | \ |
            3---4---5---6---7
            | \ | / | \ | / |
            2---3---4---5---6
            | / | \ | / | \ |
            1---2---3---4---5
            | \ | / | \ | / |
            0---1---2---3---4 > u      


            odd (u+v)%2 != 0 
           ~~~~~~~~~~~~~~~~~~~~

                  vmax
            (u,v+1)   (u+1,v+1)
              01---------11
               |       .  |
        umin   |  B  .    |   umax
               |   .      |
               | .   A    |
              00---------10
             (u,v)    (u+1,v)

                  vmin

            even   (u+v)%2 == 0 
            ~~~~~~~~~~~~~~~~~~~~~~

                   vmax
            (u,v+1)   (u+1,v+1)
              01---------11
               | .        |
       umin    |   .  D   |  umax
               |  C  .    |  
               |       .  |
              00---------10
             (u,v)    (u+1,v)
                   vmin

         */


            bool vmax_degenerate = v01 == v11 ; 
            bool vmin_degenerate = v00 == v10 ;

            bool umax_degenerate = v10 == v11 ; 
            bool umin_degenerate = v00 == v01 ;
  
            if( vmin_degenerate || vmax_degenerate ) assert( vmin_degenerate ^ vmax_degenerate ) ;
            if( umin_degenerate || umax_degenerate ) assert( umin_degenerate ^ umax_degenerate ) ;


            if( vmax_degenerate )
            {
                if(verbosity > 2)
                std::cout << "vmax_degenerate" << std::endl ; 
                add_face_( v00,v10,v11, verbosity );   // A (or C)
            } 
            else if ( vmin_degenerate )
            {
                if(verbosity > 2)
                std::cout << "vmin_degenerate" << std::endl ; 
                add_face_( v11, v01, v10, verbosity  );  // D (or B)
            }
            else if ( umin_degenerate )
            {
                if(verbosity > 2)
                std::cout << "umin_degenerate" << std::endl ; 
                add_face_( v00,v10,v11, verbosity );   // A (or D)
            }
            else if ( umax_degenerate )
            {
                if(verbosity > 2)
                std::cout << "umax_degenerate" << std::endl ; 
                add_face_( v00, v10, v01, verbosity ); // C  (or B)
            } 
            else if ((u + v) % 2)  // odd
            {
                add_face_( v00,v10,v11, verbosity  ); // A
                add_face_( v11,v01,v00, verbosity  ); // B
            } 
            else                 // even
            {
                add_face_( v00, v10, v01, verbosity  ); // C
                add_face_( v11, v01, v10, verbosity  ); // D
            }
        }
        }
    }


    int euler = euler_characteristic();
    int expect_euler = node->par_euler();
    bool euler_ok = euler == expect_euler ; 

    int nvertices = mesh.n_vertices() ;
    int expect_nvertices = node->par_nvertices(nu, nv);
    bool nvertices_ok = nvertices == expect_nvertices ; 

    if(verbosity > 0)
    {
        LOG(info) << brief() ; 
        LOG(info) << "build_parametric"
                  << " euler " << euler
                  << " expect_euler " << expect_euler
                  << ( euler_ok ? " EULER_OK " : " EULER_FAIL " )
                  << " nvertices " << nvertices
                  << " expect_nvertices " << expect_nvertices
                  << ( nvertices_ok ? " NVERTICES_OK " : " NVERTICES_FAIL " )
                  ;
        }

    if(!euler_ok || !nvertices_ok )
    {
        LOG(fatal) << "NOpenMesh::build_parametric : UNEXPECTED" ; 
        //dump("NOpenMesh::build_parametric : UNEXPECTED ");
    }

    //assert( euler_ok );
    //assert( nvertices_ok );
}



// NTriSource interface

template <typename T>
unsigned NOpenMesh<T>::get_num_tri() const
{
    return mesh.n_faces();
}
template <typename T>
unsigned NOpenMesh<T>::get_num_vert() const
{
    return mesh.n_vertices();
}

template <typename T>
void NOpenMesh<T>::get_vert( unsigned i, glm::vec3& v   ) const
{
    typedef typename T::Point          P ; 
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 

    const VH& vh = mesh.vertex_handle(i) ;
    const P& p = mesh.point(vh); 

    v.x = p[0] ; 
    v.y = p[1] ; 
    v.z = p[2] ; 
}

template <typename T>
void NOpenMesh<T>::get_tri( unsigned i, glm::uvec3& t   ) const
{
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 
    typedef typename T::ConstFaceVertexIter FVI ; 

    const FH& fh = mesh.face_handle(i) ;

    assert( mesh.valence(fh) == 3 ); 

    int n = 0 ; 
    for(FVI fv=mesh.cfv_iter(fh) ; fv.is_valid() ; fv++) 
    { 
        const VH& vh = *fv ; 
        t[n++] = vh.idx() ;
    }
    assert(n == 3);

}

template <typename T>
void NOpenMesh<T>::get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const
{
    get_tri(i, t );

    get_vert(t.x, a );
    get_vert(t.y, b );
    get_vert(t.z, c );
}


template struct NOpenMesh<NOpenMeshType> ;



