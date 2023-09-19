#pragma once
/**
U4Mesh.h : Polygonize and Serialize Solids into Triangles/Quads
==================================================================

This selects and simplifies some parts of the former x4/X4Mesh.{hh,cc} 
and it better integrates with the pyvista PolyData face format
allowing 3D visualization with quads as well as tri.  

**/

#include "G4Polyhedron.hh"
#include "NPX.h"
#include "NPFold.h"

struct U4Mesh
{
    const G4VSolid* solid ; 
    G4Polyhedron*   poly ; 
    int             nv ; 
    int             nf ; 

    NP*             vtx ; 
    double*         vtx_ ; 

    NP*             face ;  // can include tri and quad
    int*            face_ ; 

    int             nt ; 
    NP*             tri ;   // only tri 
    int*            tri_ ; 
    NP*             tpd ; 
    NP*             fpd ; 

    static NPFold* Serialize(const G4VSolid* solid) ; 
    U4Mesh(const G4VSolid* solid);     
    void init(); 
    void init_vtx(); 
    void init_face(); 
    void init_tri();
    void init_tpd();
    void init_fpd();
 
    NPFold* serialize() ; 
    std::string desc() const  ; 
};


inline NPFold* U4Mesh::Serialize(const G4VSolid* solid) // static
{
    //G4Polyhedron::SetNumberOfRotationSteps(24); 
    U4Mesh mesh(solid); 
    //G4cout << *mesh.poly << std::endl ; 
    NPFold* fold = mesh.serialize() ; 
    return fold ; 
}

inline U4Mesh::U4Mesh(const G4VSolid* solid_)
    :
    solid(solid_),
    poly(solid->CreatePolyhedron()),
    nv(poly->GetNoVertices()),
    nf(poly->GetNoFacets()),
    vtx(NP::Make<double>(nv, 3)),
    vtx_(vtx->values<double>()),
    face(NP::Make<int>(nf,4)),
    face_(face->values<int>()),
    nt(0),
    tri(nullptr),
    tri_(nullptr),
    tpd(nullptr),
    fpd(nullptr)
{
    init(); 
}

inline void U4Mesh::init()
{
    init_vtx() ; 
    init_face() ; 
    init_tri() ; 
    init_tpd() ; 
    init_fpd() ; 
}

inline void U4Mesh::init_vtx()
{
    for(int i=0 ; i < nv ; i++)
    {
        G4Point3D point = poly->GetVertex(i+1) ;  
        vtx_[3*i+0] = point.x() ; 
        vtx_[3*i+1] = point.y() ; 
        vtx_[3*i+2] = point.z() ; 
    }
}

/**
U4Mesh::init_face
-------------------

* G4Polyhedron providing 1-based vertex indices, 
  convert those to more standard 0-based 

**/

inline void U4Mesh::init_face()
{
    nt = 0 ; 
    for(int i=0 ; i < nf ; i++)
    {
        G4int nedge;
        G4int ivertex[4];
        G4int iedgeflag[4];
        G4int ifacet[4];
        poly->GetFacet(i+1, nedge, ivertex, iedgeflag, ifacet);
        assert( nedge == 3 || nedge == 4  ); 

        face_[i*4+0] = ivertex[0]-1 ;
        face_[i*4+1] = ivertex[1]-1 ;
        face_[i*4+2] = ivertex[2]-1 ;
        face_[i*4+3] = nedge == 4 ? ivertex[3]-1 : -1 ;

        switch(nedge)  // count triangles for init_tri
        {
            case 3: nt += 1 ; break ; 
            case 4: nt += 2 ; break ; 
        }
    }
}


/**
U4Mesh::init_tri
-----------------

Quads are split into two tri::

           w-------z
           |       |
           |       |
           |       |
           x-------y

           +-------z
           |     / |
           |   /   |
           | /     |
           x-------y

           w-------z
           |     / |
           |   /   |
           | /     |
           x-------+


HMM: this assumes repeatability of GetFacet ? 

* G4Polyhedron provides 1-based vertex indices, 
  convert those to much more standard 0-based 

**/

inline void U4Mesh::init_tri()
{
    assert( nt > 0 ); 
    tri = NP::Make<int>(nt, 3 ); 
    tri_ = tri->values<int>() ; 
    int jt = 0 ; 

    for(int i=0 ; i < nf ; i++)
    {
        G4int nedge;
        G4int ivertex[4];
        G4int iedgeflag[4];
        G4int ifacet[4];
        poly->GetFacet(i+1, nedge, ivertex, iedgeflag, ifacet);
        assert( nedge == 3 || nedge == 4  );

        if( nedge == 3 )
        { 
            assert( jt < nt ); 
            for(int j=0 ; j < 3 ; j++) tri_[jt*3+j] = ivertex[j]-1 ;
            jt += 1 ; 
        } 
        else if( nedge == 4 )
        {
            assert( jt < nt ); 
            for(int j=0 ; j < 3 ; j++) tri_[jt*3+j] = ivertex[j]-1 ;  // x,y,z
            jt += 1 ; 

            assert( jt < nt ); 
            tri_[jt*3+0] = ivertex[0]-1 ; // x 
            tri_[jt*3+1] = ivertex[2]-1 ; // z 
            tri_[jt*3+2] = ivertex[3]-1 ; // w 
            jt += 1 ; 
        }
    }
    assert( nt == jt ); 
}

inline void U4Mesh::init_tpd()
{
    std::vector<int> _tpd ; 
    for(int i=0 ; i < nf ; i++)
    {
        G4int nedge;
        G4int ivertex[4];
        G4int iedgeflag[4];
        G4int ifacet[4];
        poly->GetFacet(i+1, nedge, ivertex, iedgeflag, ifacet);
        assert( nedge == 3 || nedge == 4  );

        if( nedge == 3 )
        { 
            _tpd.push_back(3) ; 
            for(int j=0 ; j < 3 ; j++) _tpd.push_back(ivertex[j]-1); 
        } 
        else if( nedge == 4 )
        {
            _tpd.push_back(3) ; 
            for(int j=0 ; j < 3 ; j++) _tpd.push_back(ivertex[j]-1) ; // x,y,z

            _tpd.push_back(3) ; 
            _tpd.push_back(ivertex[0]-1);  // x
            _tpd.push_back(ivertex[2]-1);  // z
            _tpd.push_back(ivertex[3]-1);  // w
        }
    }
    tpd = NPX::Make<int>(_tpd); 
}


/**
U4Mesh::init_fpd
--------------------

The format needed by general pyvista PolyData faces 
(with tri and quad for example) requires an irregular array 
that is not easy to form from the existing regular face array.
But its trivial to form the array in C++, so here it is.
For example the faces of a quad and two triangles::

    faces = np.hstack([[4, 0, 1, 2, 3],[3, 0, 1, 4],[3, 1, 2, 4]]) 

Create pyvista PolyData from the fpd and vtx array with the below::

    pd = pv.PolyData(f.vtx, f.fpd)

For use of that in context see u4/tests/U4Mesh_test.py 


**/
inline void U4Mesh::init_fpd()
{
    std::vector<int> _fpd ; 
    for(int i=0 ; i < nf ; i++)
    {
        G4int nedge;
        G4int ivertex[4];
        G4int iedgeflag[4];
        G4int ifacet[4];
        poly->GetFacet(i+1, nedge, ivertex, iedgeflag, ifacet);
        assert( nedge == 3 || nedge == 4  ); 

        if( nedge == 3 )
        {
            _fpd.push_back(3);  
            _fpd.push_back(ivertex[0]-1);
            _fpd.push_back(ivertex[1]-1);
            _fpd.push_back(ivertex[2]-1);
        } 
        else if( nedge == 4 )
        {
            _fpd.push_back(4);  
            _fpd.push_back(ivertex[0]-1);
            _fpd.push_back(ivertex[1]-1);
            _fpd.push_back(ivertex[2]-1);
            _fpd.push_back(ivertex[3]-1);
        }
    }
    fpd = NPX::Make<int>(_fpd); 
}

inline NPFold* U4Mesh::serialize() 
{
    NPFold* fold = new NPFold ; 
    fold->add("vtx",  vtx ); 
    fold->add("face", face ); 
    fold->add("tri",  tri ); 
    fold->add("tpd",  tpd ); 
    fold->add("fpd",  fpd ); 
    return fold ; 
}

inline std::string U4Mesh::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Mesh::desc" 
       << " nv " << nv 
       << " nf " << nf 
       << std::endl 
       ; 
    std::string str = ss.str(); 
    return str ; 
}


