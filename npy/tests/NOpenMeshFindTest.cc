#include "PLOG.hh"
#include "NPY_LOG.hh"

#include "NParameters.hpp"
#include "NOpenMesh.hpp"


typedef NOpenMeshType T ; 
typedef T::VertexHandle VH ; 
typedef T::FaceHandle   FH ; 

typedef NOpenMesh<T> MESH ; 


void test_find_faces(int ctrl, NOpenMeshFindType sel)
{
    LOG(info) << "test_find_faces" ; 
    
    NParameters* meta = new NParameters ; 
    meta->add<int>("ctrl", ctrl );

    MESH* mesh = MESH::Make(NULL, meta, NULL);
    const NOpenMeshFind<T>& find = mesh->find ; 

    std::vector<FH> faces ; 
    find.find_faces(faces, sel, -1);
    find.dump_contiguity(faces);
}


void test_are_contiguous(int ctrl)
{
    LOG(info) << "test_are_contiguous" ; 

    NParameters* meta = new NParameters ; 
    meta->add<int>("ctrl", ctrl );
    meta->add<std::string>("poly", "HY" );
    meta->add<std::string>("polycfg", "sortcontiguous=0" );

    MESH* mesh = MESH::Make(NULL, meta, NULL);

    const NOpenMeshFind<T>& find = mesh->find ; 

    std::vector<FH> all_faces ; 
    find.find_faces(all_faces, FIND_ALL_FACE, -1);
    find.dump_contiguity(all_faces);
}


void test_sort_contiguous(int ctrl)
{
    LOG(info) << "test_sort_contiguous" ; 

    NParameters* meta = new NParameters ; 
    meta->add<int>("ctrl", ctrl );
    meta->add<std::string>("poly", "HY" );
    meta->add<std::string>("polycfg", "sortcontiguous=1" );

    MESH* mesh = MESH::Make(NULL, meta, NULL);


    const NOpenMeshFind<T>& find = mesh->find ; 

    std::vector<FH> faces ; 
    find.find_faces(faces, FIND_ALL_FACE, -1);

    find.sort_faces_contiguous(faces);
    //find.sort_faces_contiguous_monolithic(faces);

    find.dump_faces(faces); 

    find.dump_contiguity(faces);
}

/*
    19 verts :                                 24 faces
    12 around boundary, 7 in interior          12 with edge on boundary
                                                                     
                 9---e---8                        +---+---+           
                / \ / \ / \                      /d\c/b\a/9\
               f---3---2---d                    +---+---+---+
              / \ / \ / \ / \                  /f\e/3\2/1\8/7\
             a---4---0---1---7                +---+---+---+---+
              \ / \ / \ / \ /                  \g/h\4/5\6/n\o/
               g---5---6---i                    +---+---+---+
                \ / \ / \ /                      \i/j\k/l\m/
                 b---h---c                        +---+---+        

       it goes around the outside 

           1,8,7,o,n,m,l,k,j....c,b,a,9

       and then gets stuck at 9 
       as the remainder 6,5,4,3,2 are not connected to the cursor... need to backtrack when get stuck
*/
 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    static const int N = 2 ; 
    int ctrl[N] = {3, 666 } ;
    std::string name[N] = {"tripatch", "hexpatch" } ;


    for(unsigned i=1 ; i < N ; i++)
    {
        LOG(info) << " ctrl " << ctrl[i] << " " << name[i]  ; 
        if(ctrl[i] == 3)   std::cout << NOpenMeshConst::TRIPATCH << std::endl ; 
        if(ctrl[i] == 666) std::cout << NOpenMeshConst::HEXPATCH << std::endl ; 

        //test_are_contiguous(ctrl[i]); 
        test_sort_contiguous(ctrl[i]); 
        //test_find_faces(ctrl[i], FIND_SIDECORNER_FACE); 
    }

    return 0 ; 
}
  
  
