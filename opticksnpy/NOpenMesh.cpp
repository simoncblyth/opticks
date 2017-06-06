//  grab lower dependency pieces from openmeshrap MWrap.cc as needed


#include <limits>
#include <iostream>
#include <sstream>


#include "PLOG.hh"
#include "NGLM.hpp"
#include "Nuv.hpp"
#include "NNode.hpp"
#include "NCSGBSP.hpp"

#include "NOpenMesh.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"

#include <OpenMesh/Tools/Utils/MeshCheckerT.hh>


template <typename T>
const char* NOpenMesh<T>::COMBINE_HYBRID_ = "COMBINE_HYBRID" ; 

template <typename T>
const char* NOpenMesh<T>::COMBINE_CSGBSP_ = "COMBINE_CSGBSP" ; 

template <typename T>
const char* NOpenMesh<T>::MeshModeString(NOpenMeshMode_t meshmode)
{
    const char* s = NULL ; 
    if(     meshmode & COMBINE_HYBRID) s = COMBINE_HYBRID_  ;
    else if(meshmode & COMBINE_CSGBSP) s = COMBINE_CSGBSP_  ;
    return s ; 
}


template <typename T>
NOpenMesh<T>* NOpenMesh<T>::spawn_left()
{
    return new NOpenMesh<T>(node->left, level, verbosity, ctrl, meshmode, epsilon ) ;
}
template <typename T>
NOpenMesh<T>* NOpenMesh<T>::spawn_right()
{
    return new NOpenMesh<T>(node->right, level, verbosity, ctrl, meshmode, epsilon ) ;
}


 

template <typename T>
NOpenMesh<T>::NOpenMesh(const nnode* node, int level, int verbosity, int ctrl, NOpenMeshMode_t meshmode, float epsilon)
    :
    prop(mesh),
    desc(mesh, prop),
    find(mesh, prop, verbosity),
    build(mesh, prop, desc, find, verbosity),
    subdiv(mesh, prop, desc, find, build, verbosity, epsilon ),

    node(node), 
    level(level), 
    verbosity(verbosity), 
    ctrl(ctrl), 
    meshmode(meshmode),
    epsilon(epsilon),
    leftmesh(NULL),
    rightmesh(NULL)
{
    init();
}


template <typename T>
void NOpenMesh<T>::init()
{
    if(!node) return ; 

    LOG(info) << "NOpenMesh<T>::init()"
              << " meshmode " << meshmode 
              << " MeshModeString " << MeshModeString(meshmode) 
              ; 

    build_csg();
    check();
}


template <typename T>
void NOpenMesh<T>::check()
{
    assert(OpenMesh::Utils::MeshCheckerT<T>(mesh).check()) ;
    LOG(info) << "NOpenMesh<T>::check OK" ; 
}



template <typename T>
void NOpenMesh<T>::one_subdiv(NOpenMeshFindType select, int param, const nnode* other)
{
    std::vector<FH> target_faces ; 
    find.find_faces( target_faces, select,  param );

    unsigned n_target_faces = target_faces.size() ;

    LOG(info) << "one_subdiv" 
              << " ctrl " << ctrl 
              << " verbosity " << verbosity
              << " n_target_faces " << n_target_faces
              << desc.desc_euler()
              ;

    int depth = 0 ; 
    for(unsigned i=0 ; i < n_target_faces ; i++) 
    {
        FH fh = target_faces[i] ;
        subdiv.sqrt3_split_r(fh,  other, depth );
    }
    mesh.garbage_collection();  // NB this invalidates handles, so dont hold on to them
}
 



template <typename T>
void NOpenMesh<T>::subdiv_test()
{
    unsigned nloop0 = find.find_boundary_loops() ;
    LOG(info) << "subdiv_test START " 
              << " ctrl " << ctrl 
              << " verbosity " << verbosity
              << " nloop0 " << nloop0
              << desc.desc_euler()
              ;
    if(verbosity > 3) std::cout << desc.faces() << std::endl ;  

    const nnode* other = NULL ; 

    one_subdiv(FIND_ALL_FACE, -1, other) ;
    //one_subdiv(FIND_NONBOUNDARY_FACE, -1, other);
    //one_subdiv(FIND_FACEMASK_FACE, -1, other);

    //one_subdiv(FIND_BOUNDARY_FACE, -1, other);
    //one_subdiv(FIND_NONBOUNDARY_FACE, -1, other);

    //one_subdiv(FIND_IDENTITY_FACE, 101 , other);
    //one_subdiv(FIND_IDENTITY_FACE, 2, other);
    //one_subdiv(FIND_IDENTITY_FACE,  101, other);
 

    unsigned nloop1 = find.find_boundary_loops() ;
    LOG(info) << "subdiv_test DONE " 
              << " ctrl " << ctrl 
              << " verbosity " << verbosity
              << " nloop1 " << nloop1
              << desc.desc_euler()
              ;

    if(verbosity > 3) std::cout << desc.faces() << std::endl ;  
}





template <typename T>
void NOpenMesh<T>::build_csg()
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
        build.add_parametric_primitive(node, level, ctrl, epsilon ); // adds unique vertices and faces to build out the parametric mesh  
        build.euler_check(node, level);
    }
    else
    {
        assert(node->type == CSG_UNION || node->type == CSG_INTERSECTION || node->type == CSG_DIFFERENCE );
        LOG(info) 
                  << "NOpenMesh<T>::build_csg" 
                  << " making leftmesh rightmesh "
                  << " meshmode " << meshmode
                  << " COMBINE_HYBRID " << COMBINE_HYBRID
                  << " COMBINE_CSGBSP " << COMBINE_CSGBSP
                  << " MeshModeString " << MeshModeString(meshmode)
                  ; 

        leftmesh = spawn_left() ; 
        rightmesh = spawn_right() ; 


        LOG(info) 
                  << "NOpenMesh<T>::build_csg" 
                  << " START "
                  ;

        std::cout << " leftmesh  " << leftmesh->brief() << std::endl ; 
        std::cout << " rightmesh " << rightmesh->brief() << std::endl ; 


        if( meshmode & COMBINE_HYBRID )
        {
            combine_hybrid();
        }
        else if( meshmode & COMBINE_CSGBSP )
        {
            combine_csgbsp() ; 
        }
        else
        {
            assert(0) ; 
        }

        LOG(info) 
                  << "NOpenMesh<T>::build_csg" 
                  << " DONE "
                  ;

        std::cout << " leftmesh  " << leftmesh->brief() << std::endl ; 
        std::cout << " rightmesh " << rightmesh->brief() << std::endl ; 
        std::cout << " combined " << brief() << std::endl ; 
    }
}



template <typename T>
void NOpenMesh<T>::combine_csgbsp()
{
    NCSGBSP csgbsp( leftmesh, rightmesh, node->type );
    build.copy_faces( &csgbsp, epsilon );
}


template <typename T>
void NOpenMesh<T>::combine_hybrid( )
{
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

    LOG(info) << "combine_hybrid" 
              << " leftmesh " << leftmesh
              << " rightmesh " << rightmesh
               ;

    // initial meshes should be closed with no boundary loops
    assert( leftmesh->find.find_boundary_loops() == 0) ;   
    assert( rightmesh->find.find_boundary_loops() == 0) ;  

    leftmesh->build.mark_faces( node->right );
    rightmesh->build.mark_faces( node->left );

    leftmesh->one_subdiv( FIND_FACEMASK_FACE, -1, node->right);
    rightmesh->one_subdiv( FIND_FACEMASK_FACE, -1, node->left);

    leftmesh->build.mark_faces( node->right );
    rightmesh->build.mark_faces( node->left );


    // despite subdiv should still be closed zero boundary meshes
    assert( leftmesh->find.find_boundary_loops() == 0) ;   
    assert( rightmesh->find.find_boundary_loops() == 0) ;  


    if(node->type == CSG_UNION)
    {
        build.copy_faces( leftmesh,  NOpenMeshProp<T>::ALL_OUTSIDE_OTHER, epsilon );
        build.copy_faces( rightmesh, NOpenMeshProp<T>::ALL_OUTSIDE_OTHER, epsilon );
    }
    else if(node->type == CSG_INTERSECTION)
    {
        build.copy_faces( leftmesh,  NOpenMeshProp<T>::ALL_INSIDE_OTHER, epsilon  );
        build.copy_faces( rightmesh, NOpenMeshProp<T>::ALL_INSIDE_OTHER, epsilon  );
    }
    else if(node->type == CSG_DIFFERENCE )
    {
        build.copy_faces( leftmesh,  NOpenMeshProp<T>::ALL_OUTSIDE_OTHER, epsilon  );
        build.copy_faces( rightmesh, NOpenMeshProp<T>::ALL_INSIDE_OTHER, epsilon  );
    }


    int nloop = find.find_boundary_loops() ;
    // hmm expecting 2, but thats geometry specific

    if(verbosity > 0)
    LOG(info) << "combine_hybrid"
              << " boundary_loops " << nloop 
               ;    


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


    for( FI f=a_mesh->mesh.faces_begin() ; f != a_mesh->mesh.faces_end(); ++f ) 
    {
        const FH fh = *f ;  
        if(!prop.is_facemask_face(fh, -1)) continue ; 
            
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

            uv[0] = a_mesh->prop.get_uv(vh[0]) ; 
            uv[1] = a_mesh->prop.get_uv(vh[1]) ; 

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
    LOG(info) << "dump START" ; 
    LOG(info) << msg << " " << brief() ; 
    desc.dump_vertices();
    desc.dump_faces();
    LOG(info) << "dump DONE" ; 
}


template <typename T>
std::string NOpenMesh<T>::brief()
{
    return desc.desc_euler();
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
    const VH vh = mesh.vertex_handle(i) ;
    const P  p = mesh.point(vh); 

    v.x = p[0] ; 
    v.y = p[1] ; 
    v.z = p[2] ; 
}

template <typename T>
void NOpenMesh<T>::get_normal( unsigned i, glm::vec3& n   ) const
{
    const VH vh = mesh.vertex_handle(i) ;
    const P  nrm = mesh.normal(vh); 

    n.x = nrm[0] ; 
    n.y = nrm[1] ; 
    n.z = nrm[2] ; 
}



template <typename T>
void NOpenMesh<T>::get_uv( unsigned i, glm::vec3& ouv   ) const
{
    const VH vh = mesh.vertex_handle(i) ;
    nuv uv = prop.get_uv( vh );

    ouv.x = uv.u() ;   // <-- currently the integer values
    ouv.y = uv.v() ; 
    ouv.z = uv.s() ; 
}


template <typename T>
void NOpenMesh<T>::get_tri( unsigned i, glm::uvec3& t   ) const
{
    //typedef typename T::VertexHandle   VH ; 
    //typedef typename T::FaceHandle     FH ; 
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


// debug shapes

template <typename T>
NOpenMesh<T>* NOpenMesh<T>::cube(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    m->build.add_cube();
    m->check();
    return m ; 
}
 
template <typename T>
NOpenMesh<T>* NOpenMesh<T>::tetrahedron(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    m->build.add_tetrahedron();
    //std::cout << m->desc.desc() << std::endl ; 
    m->check();
    return m ; 
}

template <typename T>
NOpenMesh<T>* NOpenMesh<T>::hexpatch(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    bool inner_only = false ; 
    m->build.add_hexpatch(inner_only );
    m->check();
    return m ; 
}

template <typename T>
NOpenMesh<T>* NOpenMesh<T>::hexpatch_inner(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    bool inner_only = true ; 
    m->build.add_hexpatch(inner_only );
    m->check();
    return m ; 
}


template struct NOpenMesh<NOpenMeshType> ;


