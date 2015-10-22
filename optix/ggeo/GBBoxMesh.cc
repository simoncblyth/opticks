#include "GBBoxMesh.hh"

#define DEBUG 1 
#ifdef DEBUG
#include <sstream>
#include <iostream>
#include <iomanip>
#endif


GBBoxMesh* GBBoxMesh::create(GMergedMesh* mergedmesh)
{
    GBBoxMesh* bbm = new GBBoxMesh(mergedmesh);
    return bbm ;
}



GBBoxMesh::GBBoxMesh(GMergedMesh* mergedmesh)
       : 
       m_mergedmesh(mergedmesh),
       GMesh(mergedmesh->getIndex(),
             new gfloat3[NUM_VERTICES],    // vertices
             NUM_VERTICES,  
             new guint3[NUM_FACES],        // faces
             NUM_FACES,    
             new gfloat3[NUM_VERTICES],    // normals
             NULL )                        // texcoords
     // backing buffers are created by the GMesh ctor
{
      twentyfour();

      setColors(  new gfloat3[NUM_VERTICES]);
      setColor(0.5,0.5,0.5);  

      // needed for instanced : but this is stealing 
     setITransformsBuffer(mergedmesh->getITransformsBuffer());

     setInstanceSlice(mergedmesh->getInstanceSlice());
     //setFaceSlice(mergedmesh->getFaceSlice());
     setFaceSlice(NULL);   // dont pass along the face slicing

}


void GBBoxMesh::twentyfour()
{
   /*
      Bounding boxes look OK with filled rendering, 
      wireframe rendering has cosmetic problems: need to 
      rearrange the ordering of the triangles to get a 
      proper wireframe...
 
      However wireframe is generally slow compared to filled
      and the whole point of using bbox is speed so 
      currently just using the easier to get right filled triangle rendering.
   */


     gbbox bb = m_mergedmesh->getBBox(0);
     //bb.Summary("GBBoxMesh::twentyfour");


            /*

      +y         Axis Aligned Box
      |
      |            "By convention, polygons whose vertices appear 
      0---- +x      in counterclockwise order on the screen are called front-facing."
     /
    /
   +z     

            */


     enum { kPX, kPY, kPZ, kMX, kMY, kMZ, kFace } ;

     unsigned int nv(0) ;
     unsigned int nf(0) ;

     for(unsigned int face=0 ; face < kFace ; face++)   // over 6 faces of the bbox
     {

         switch(face)
         {                        
             case kPX:
                /*
                      +y
                      | 
                      |
                      | 
               +z-----0                        
                        
                      (+x out of page)                       
                           
                 1            0 
                    *-------* 
                    |       |
                    |  +X   |
                    |       |
                    *-------*   
                 2            3
 
                 */
                       m_vertices[nv+0] = gfloat3( bb.max.x, bb.max.y, bb.min.z ) ; 
                       m_vertices[nv+1] = gfloat3( bb.max.x, bb.max.y, bb.max.z ) ; 
                       m_vertices[nv+2] = gfloat3( bb.max.x, bb.min.y, bb.max.z ) ; 
                       m_vertices[nv+3] = gfloat3( bb.max.x, bb.min.y, bb.min.z ) ; 
                       for(unsigned int i=0 ; i < 4 ; i++) m_normals[nv+i]  = gfloat3( 1.f, 0.f, 0.f );
                       break;
             case kMX:
             /* 

                      +y
                      | 
                      |
                      | 
                      0-----+z          

                  (-x out of page)                       
                           
                 1            0 
                    *-------* 
                    |       |
                    |  -X   |
                    |       |
                    *-------*   
                 2            3

             */

                       m_vertices[nv+0] = gfloat3( bb.min.x, bb.max.y, bb.max.z ) ; 
                       m_vertices[nv+1] = gfloat3( bb.min.x, bb.max.y, bb.min.z ) ; 
                       m_vertices[nv+2] = gfloat3( bb.min.x, bb.min.y, bb.min.z ) ; 
                       m_vertices[nv+3] = gfloat3( bb.min.x, bb.min.y, bb.max.z ) ; 
                       for(unsigned int i=0 ; i < 4 ; i++) m_normals[nv+i]  = gfloat3( -1.f, 0.f, 0.f );
                       break;
            
             case kPY:

                /*

                      (+y out of page)                       

                      0----- +x
                      |
                      |
                      |
                      +z                        
                           
                 1            0 
                    *-------* 
                    |       |
                    |  +Y   |
                    |       |
                    *-------*   
                 2            3
 
                 */
                       m_vertices[nv+0] = gfloat3( bb.max.x, bb.max.y, bb.min.z ) ; 
                       m_vertices[nv+1] = gfloat3( bb.min.x, bb.max.y, bb.min.z ) ; 
                       m_vertices[nv+2] = gfloat3( bb.min.x, bb.max.y, bb.max.z ) ; 
                       m_vertices[nv+3] = gfloat3( bb.max.x, bb.max.y, bb.max.z ) ; 
                       for(unsigned int i=0 ; i < 4 ; i++) m_normals[nv+i]  = gfloat3( 0.f, 1.f, 0.f );
                       break;
             case kMY:
             /* 

                      +z
                      | 
                      |
                      | 
                      0-----+x          

                  (-y out of page)                       
                           
                 1            0 
                    *-------* 
                    |       |
                    |  -Y   |
                    |       |
                    *-------*   
                 2            3

             */

                       m_vertices[nv+0] = gfloat3( bb.min.x, bb.min.y, bb.max.z ) ; 
                       m_vertices[nv+1] = gfloat3( bb.max.x, bb.min.y, bb.max.z ) ; 
                       m_vertices[nv+2] = gfloat3( bb.max.x, bb.min.y, bb.min.z ) ; 
                       m_vertices[nv+3] = gfloat3( bb.min.x, bb.min.y, bb.min.z ) ; 
                       for(unsigned int i=0 ; i < 4 ; i++) m_normals[nv+i]  = gfloat3( 0.f, -1.f, 0.f );
                       break;

             case kPZ:

                /*
                      (+z out of page)                       

                     +y
                      | 
                      |
                      |
                      0----- +x
                           
                 1            0 
                    *-------* 
                    |       |
                    |  +Z   |
                    |       |
                    *-------*   
                 2            3
 
                 */
                       m_vertices[nv+0] = gfloat3( bb.max.x, bb.max.y, bb.max.z ) ; 
                       m_vertices[nv+1] = gfloat3( bb.min.x, bb.max.y, bb.max.z ) ; 
                       m_vertices[nv+2] = gfloat3( bb.min.x, bb.min.y, bb.max.z ) ; 
                       m_vertices[nv+3] = gfloat3( bb.max.x, bb.min.y, bb.max.z ) ; 
                       for(unsigned int i=0 ; i < 4 ; i++) m_normals[nv+i]  = gfloat3( 0.f, 0.f, 1.f );
                       break;

             case kMZ:

                /*
                      (-z out of page)                       

                     +y
                      | 
                      |
                      |
              +x------0
                
                 1            0 
                    *-------* 
                    |       |
                    |  -Z   |
                    |       |
                    *-------*   
                 2            3
 
                 */
                       m_vertices[nv+0] = gfloat3( bb.min.x, bb.max.y, bb.min.z ) ; 
                       m_vertices[nv+1] = gfloat3( bb.max.x, bb.max.y, bb.min.z ) ; 
                       m_vertices[nv+2] = gfloat3( bb.max.x, bb.min.y, bb.min.z ) ; 
                       m_vertices[nv+3] = gfloat3( bb.min.x, bb.min.y, bb.min.z ) ; 
                       for(unsigned int i=0 ; i < 4 ; i++) m_normals[nv+i]  = gfloat3( 0.f, 0.f, -1.f );
                       break;
         }



         m_faces[nf+0] = guint3(nv+0, nv+1, nv+2 );
         m_faces[nf+1] = guint3(nv+2, nv+3, nv+0 ); 

         nv += 4 ; 
         nf += 2 ; 
     }  

    
}





void GBBoxMesh::eight()
{

     gbbox bb = m_mergedmesh->getBBox(0);
     bb.Summary("GBBoxMesh::init");

     // from min corner (nearest origin) ascend to max along one axis at a time
     gfloat3 ___(bb.min.x,  bb.min.y, bb.min.z) ;
     gfloat3 x__(bb.max.x,  bb.min.y, bb.min.z) ; 
     gfloat3 _y_(bb.min.x,  bb.max.y, bb.min.z) ; 
     gfloat3 __z(bb.min.x,  bb.min.y, bb.max.z) ; 

     // from max corner (furthest from origin) descend to min along one axis at a time
     gfloat3 xyz(bb.max.x,  bb.max.y, bb.max.z) ;
     gfloat3 _yz(bb.min.x,  bb.max.y, bb.max.z) ;
     gfloat3 x_z(bb.max.x,  bb.min.y, bb.max.z) ;
     gfloat3 xy_(bb.max.x,  bb.max.y, bb.min.z) ;


/*
      +y         Axis Aligned Box
      |
      |            "By convention, polygons whose vertices appear 
      0---- +x      in counterclockwise order on the screen are called front-facing."
     /
    /
   +z     

       [5:_y_] . . . .[6:xy_]
         /.           /.
        / .          / .
       /  .         /  . 
   [1:_yz]--------[0:xyz] 
      |   .        |   .
      | [4:___]. . | [7:x__]
      |  /         |  /  
      | /          | / 
      |/           |/
      +------------+
   [2:__z]         [3:x_z]

*/

     
     m_vertices[0] = xyz  ; 
     m_vertices[1] = _yz  ; 
     m_vertices[2] = __z  ; 
     m_vertices[3] = x_z  ;

     m_vertices[4] = ___  ;
     m_vertices[5] = _y_  ;
     m_vertices[6] = xy_  ;
     m_vertices[7] = x__  ;
    

     m_faces[0] = guint3(0, 1, 2 );   // front  (+z)
     m_faces[1] = guint3(2, 3, 0 ); 

     m_faces[2] = guint3(2, 1, 5 );   // left   (-x)
     m_faces[3] = guint3(5, 4, 2 ); 

     m_faces[4] = guint3(2, 4, 7 );   // lower  (-y)
     m_faces[5] = guint3(7, 3, 2 );
 
     m_faces[6] = guint3(0, 3, 7 );   // right (+x)
     m_faces[7] = guint3(7, 6, 0 );
 
     m_faces[8] = guint3(4, 5, 6 );   // back (-z)
     m_faces[9] = guint3(6, 7, 4 );
 
     m_faces[10] = guint3(1, 0, 6 );   // upper (+y)
     m_faces[11] = guint3(6, 5, 1 );

     // http://stackoverflow.com/questions/6656358/calculating-normals-in-a-triangle-mesh/6661242#6661242
     // hmm vertex normals only defined by averaging over adjacent edges ??
     //  ... move to face normals ?
     //
     // http://stackoverflow.com/questions/5046579/vertex-normals-for-a-cube
     //

}



