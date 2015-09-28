//  file:///usr/local/env/graphics/openmesh/OpenMesh-4.1/Documentation/a00044.html

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <string>


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;




#include "md5digest.hpp"
#include "NPY.hpp"


#include <OpenMesh/Core/IO/MeshIO.hh>
//#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
//typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;
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
    Ary(T* data, unsigned int num , unsigned int elem) : data(data), num(num), elem(elem) {} ;

    T* data ; 
    unsigned int num  ; 
    unsigned int elem ; 
};



template <typename MeshT>
inline MeshT* convertToOpenMesh(char* dir)
{
   // developed with single/few mesh caches in mind like --jdyb --kdyb 

   Cache cache(dir); 

   NPY<float>* vertices_ = NPY<float>::load( cache.path("GMergedMesh/0/vertices.npy").c_str() ); 
   NPY<int>* faces_      = NPY<int>::load( cache.path("GMergedMesh/0/indices.npy").c_str() ); 

   Ary<float> vertices( vertices_->getValues(), vertices_->getShape(0), 3 );
   Ary<int>   faces(       faces_->getValues(), faces_->getShape(0)/3, 3 ); // at NPY level indices have shape (3n, 1) rather than (n,3)

   // de-duping vertices is mandatory otherwise mesh topology cannot be accessed

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

       printf(" %4d : %4d : %10.3f %10.3f %10.3f : %s \n", i, o2n[i], 
                 *(vertices.data+3*i+0), *(vertices.data+3*i+1), *(vertices.data+3*i+2), dig.c_str() );
   }

   Ary<float> dd_vertices( new float[vidx*3], vidx , 3 );
   Ary<int>   dd_faces( new int[faces.num*3], faces.num , 3 );

   // copy old vertices to new leaving out the dupes ... 

   for(int n=0 ; n < vidx ; ++n )
   {
       int o = n2o[n] ;
       printf(" n %4d n2o %4d \n", n, o );

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
   


   MeshT* mesh = new MeshT ;
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
   int* fdata = dd_faces.data ;
   assert(dd_faces.elem == 3); 
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
   return mesh ; 
}



template <typename MeshT>
inline void labelConnectedComponents(MeshT* mesh)
{
    typedef typename MeshT::VertexHandle VH ;
    typedef typename MeshT::VertexIter VI ; 
    typedef typename MeshT::VertexVertexIter VVI ;

    OpenMesh::VPropHandleT<int> component;
    mesh->add_property(component); 

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


    for( VI vi=mesh->vertices_begin() ; vi != mesh->vertices_end(); ++vi )
         std::cout << " vit " <<  *vi
                   << " point " << mesh->point(*vi) 
                   << " comp " << mesh->property(component, *vi) 
                   << std::endl ;
}



int main()
{

    MyMesh* mesh = convertToOpenMesh<MyMesh>(getenv("JDPATH"));

    mesh->request_face_normals();
    mesh->update_normals();

    labelConnectedComponents<MyMesh>(mesh); 


  //  cf http://www.openflipper.org/media/Documentation/OpenFlipper-1.0.2/MeshInfoT_8cc_source.html




  MyMesh::FaceIter fit ; 
  for (fit=mesh->faces_begin(); fit!=mesh->faces_end(); ++fit)
  {
      std::cout << " fit " << *fit 
                << std::endl;
  } 
  





  // write mesh to output.obj
  try
  {
    if ( !OpenMesh::IO::write_mesh(*mesh, "/tmp/output.off") )
    {
      std::cerr << "Cannot write mesh to file '/tmp/output.off'" << std::endl;
      return 1;
    }
  }
  catch( std::exception& x )
  {
    std::cerr << x.what() << std::endl;
    return 1;
  }
  return 0;
}


