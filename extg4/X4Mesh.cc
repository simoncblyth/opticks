#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "X4Mesh.hh"
#include "G4VSolid.hh"
#include "G4Polyhedron.hh"
#include "SDirect.hh"
#include "NPY.hpp"
#include "PLOG.hh"


X4Mesh::X4Mesh( const G4VSolid* solid ) 
   :
   m_solid(solid),
   m_polyhedron(NULL),
   m_vtx(NULL),
   m_raw(NULL),
   m_tri(NULL),
   m_verbosity(0)
{
   init() ;
}

void X4Mesh::init()
{
    polygonize();
    collect();
}

std::string X4Mesh::desc() const 
{
    std::stringstream ss ; 
    ss << "X4Mesh"
       << " vtx " << ( m_vtx ? m_vtx->getShapeString() : "-" )
       << " raw " << ( m_raw ? m_raw->getShapeString() : "-" )
       << " tri " << ( m_tri ? m_tri->getShapeString() : "-" )
       ;
    return ss.str();
}

void X4Mesh::polygonize()
{
    G4bool create = true ; 
    G4int noofsides = 24 ;
    G4Polyhedron::SetNumberOfRotationSteps(noofsides);

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());
       if( create ){         
           m_polyhedron = m_solid->CreatePolyhedron ();  // always create a new poly   
       } else {
           m_polyhedron = m_solid->GetPolyhedron ();     // if poly created already and no parameter change just provide that one 
       }   
    }   

    LOG(info) << " cout " << coutbuf.str() ;
    LOG(info) << " cerr " << cerrbuf.str() ;
    
    std::string polysmry ; 
    { 
       std::stringstream ss ;
       ss << "v " << m_polyhedron->GetNoVertices() << " " ;
       ss << "f " << m_polyhedron->GetNoFacets() << " " ;
       polysmry = ss.str();
    }
   
    LOG(info) << polysmry ; 
}


void X4Mesh::collect()
{
    G4int nv = m_polyhedron->GetNoVertices();
    G4int nf = m_polyhedron->GetNoFacets();

    m_vtx = NPY<float>::make(nv, 4) ; 
    m_vtx->zero();

    m_raw = NPY<unsigned>::make(nf, 4) ; 
    m_raw->zero();

    for (int i = 0 ; i < nv ; i++) collect_vtx(i+1) ; 
    for (int i = 0 ; i < nf ; i++) collect_raw(i+1) ; 

    collect_tri(); 

    m_vtx->save("/tmp/X4Mesh/vtx.npy"); 
    m_raw->save("/tmp/X4Mesh/raw.npy"); 
    m_tri->save("/tmp/X4Mesh/tri.npy"); 
}


void X4Mesh::collect_vtx(int ivert)
{
    G4Point3D vtx = m_polyhedron->GetVertex(ivert); // ivert is 1-based index from 1 to nv
    m_vtx->setQuad( ivert-1, 0, 0,   vtx.x(), vtx.y(), vtx.z(), 1.f );    
}

void X4Mesh::collect_raw(int iface)
{
    G4int nedge;
    G4int ivertex[4];
    G4int iedgeflag[4];
    G4int ifacet[4];

    m_polyhedron->GetFacet(iface, nedge, ivertex, iedgeflag, ifacet);

    if(m_verbosity > 1)
    {
        std::cout 
            << " iface - 1 " << iface - 1 
            << " nedge " << nedge 
            << " verts ( " 
            ;
    }

    assert( nedge == 3 || nedge == 4 ); 

    if(m_verbosity > 1 )
    {
        G4int iedge = 0;
        for(iedge = 0; iedge < nedge; ++iedge) {
            std::cout << ivertex[iedge] - 1 << " " ;   
        }   
        std::cout << " ) " << std::endl ; 
    }


    // collect raw indices, allowing tris or quads 
    // vertex indices are 1-based, for tris the 4th slot is zero 

    m_raw->setQuad( iface-1, 0, 0, 
            ivertex[0] ,
            ivertex[1] , 
            ivertex[2] ,
            nedge == 4 ? ivertex[3]  : 0 ) ;
}


void X4Mesh::collect_tri()
{
    unsigned nf = m_raw->getShape(0);  

    unsigned ntri = 0 ; 
    for(unsigned i=0 ; i < nf ; i++)
    {
         glm::uvec4 raw = m_raw->getQuadU(i, 0) ; 
         ntri += ( raw.w == 0 ? 1 : 2 ) ; 
    }

    // 4 if for standards/alignment purpose, not to hold anything in 4th slot 
    m_tri = NPY<unsigned>::make( ntri, 4 ) ;  
    m_tri->zero();

    unsigned jtri = 0 ; 
    for(unsigned i=0 ; i < nf ; i++)
    {
         glm::uvec4 raw = m_raw->getQuadU(i, 0) ; 

/*
         std::cout 
             << std::setw(6) << raw.x   
             << std::setw(6) << raw.y
             << std::setw(6) << raw.z   
             << std::setw(6) << raw.w
             << std::endl ;    
*/
         if(raw.w == 0)
         {
             m_tri->setQuad( jtri, 0, 0,   raw.x-1, raw.y-1, raw.z-1, 0 ) ;
             jtri += 1 ;  
         } 
         else if ( raw.w != 0 )
         {
             m_tri->setQuad( jtri, 0, 0,   raw.x-1, raw.y-1, raw.z-1, 0 ) ;
             jtri += 1 ;  

             m_tri->setQuad( jtri, 0, 0,   raw.x-1, raw.z-1, raw.w-1, 0 ) ;
             jtri += 1 ;  
         }
    }

    assert( jtri == ntri );

    /*

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

    */
}





