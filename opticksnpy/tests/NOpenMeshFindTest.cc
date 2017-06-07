#include "PLOG.hh"
#include "NPY_LOG.hh"

#include "NOpenMesh.hpp"


typedef NOpenMeshType T ; 
typedef T::VertexHandle VH ; 
typedef T::FaceHandle   FH ; 




void test_are_contiguous(int ctrl)
{
    LOG(info) << "test_are_contiguous" ; 
    int level = 0 ; 
    int verbosity = 3 ; 
    const char* polycfg = "sortcontiguous=0" ; 

    NOpenMesh<T>* mesh = NOpenMesh<T>::BuildTest(level, verbosity, ctrl, polycfg  );
    const NOpenMeshFind<T>& find = mesh->find ; 

    std::vector<FH> all_faces ; 
    find.find_faces(all_faces, FIND_ALL_FACE, -1);
    find.dump_contiguity(all_faces);
}


void test_sort_contiguous(int ctrl)
{
    LOG(info) << "test_sort_contiguous" ; 
    int level = 0 ; 
    int verbosity = 3 ; 
    const char* polycfg = "sortcontiguous=1" ; 

    NOpenMesh<T>* mesh = NOpenMesh<T>::BuildTest(level, verbosity, ctrl, polycfg  );
    const NOpenMeshFind<T>& find = mesh->find ; 

    std::vector<FH> all_faces ; 
    find.find_faces(all_faces, FIND_ALL_FACE, -1);
    find.dump_contiguity(all_faces);
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

   hmm.. 

       it goes around the outside 

           1,8,7,o,n,m,l,k,j....c,b,a,9

       and then gets stuck at 9 
       as the remainder 6,5,4,3,2 are not 



enum { A=10, B=11, C=12, D=13, E=14, F=15, G=16, H=17, I=18, J=19, K=20, L=21, M=22, N=23, O=24 } ;

017-06-07 21:00:18.380 INFO  [5782657] [>::find_faces@94] NOpenMeshFind<T>::find_faces   FindType FIND_ALL_FACE param -1 count 24 totface 24 cfg.reversed 0
 q  23 c  25 cursor      0 candidate  23 cursor_id      1 candidate_id  23 idc   n
 q  23 c  25 cursor      0 candidate  22 cursor_id      1 candidate_id  22 idc   m
 q  23 c  25 cursor      0 candidate  21 cursor_id      1 candidate_id  21 idc   l
 q  23 c  25 cursor      0 candidate  20 cursor_id      1 candidate_id  20 idc   k
 q  23 c  25 cursor      0 candidate  19 cursor_id      1 candidate_id  19 idc   j
 q  23 c  25 cursor      0 candidate  18 cursor_id      1 candidate_id  18 idc   i
 q  23 c  25 cursor      0 candidate  17 cursor_id      1 candidate_id  17 idc   h
 q  23 c  25 cursor      0 candidate  16 cursor_id      1 candidate_id  16 idc   g
 q  23 c  25 cursor      0 candidate  15 cursor_id      1 candidate_id  15 idc   f
 q  23 c  25 cursor      0 candidate  14 cursor_id      1 candidate_id  14 idc   e
 q  23 c  25 cursor      0 candidate  13 cursor_id      1 candidate_id  13 idc   d
 q  23 c  25 cursor      0 candidate  12 cursor_id      1 candidate_id  12 idc   c
 q  23 c  25 cursor      0 candidate  11 cursor_id      1 candidate_id  11 idc   b
 q  23 c  25 cursor      0 candidate  10 cursor_id      1 candidate_id  10 idc   a
 q  23 c  25 cursor      0 candidate   9 cursor_id      1 candidate_id   9 idc   9
 q  23 c  25 cursor      0 candidate   8 cursor_id      1 candidate_id   8 idc   8 is_contiguous 
 q  22 c  26 cursor      8 candidate   7 cursor_id      8 candidate_id   7 idc   7 is_contiguous 
 q  21 c  27 cursor      7 candidate   6 cursor_id      7 candidate_id  24 idc   o is_contiguous 
 q  20 c  28 cursor      6 candidate   5 cursor_id     24 candidate_id   6 idc   6
 q  20 c  28 cursor      6 candidate   4 cursor_id     24 candidate_id   5 idc   5
 q  20 c  28 cursor      6 candidate   3 cursor_id     24 candidate_id   4 idc   4
 q  20 c  28 cursor      6 candidate   2 cursor_id     24 candidate_id   3 idc   3
 q  20 c  28 cursor      6 candidate   1 cursor_id     24 candidate_id   2 idc   2
 q  20 c  28 cursor      6 candidate   0 cursor_id     24 candidate_id   1 idc   1
 q  20 c  28 cursor      6 candidate  23 cursor_id     24 candidate_id  23 idc   n is_contiguous 
 q  19 c  29 cursor     23 candidate  22 cursor_id     23 candidate_id  22 idc   m is_contiguous 
 q  18 c  30 cursor     22 candidate  21 cursor_id     22 candidate_id  21 idc   l is_contiguous 
 q  17 c  31 cursor     21 candidate  20 cursor_id     21 candidate_id  20 idc   k is_contiguous 
 q  16 c  32 cursor     20 candidate  19 cursor_id     20 candidate_id  19 idc   j is_contiguous 
 q  15 c  33 cursor     19 candidate  18 cursor_id     19 candidate_id  18 idc   i is_contiguous 
 q  14 c  34 cursor     18 candidate  17 cursor_id     18 candidate_id  17 idc   h is_contiguous 
 q  13 c  35 cursor     17 candidate  16 cursor_id     17 candidate_id  16 idc   g is_contiguous 
 q  12 c  36 cursor     16 candidate  15 cursor_id     16 candidate_id  15 idc   f is_contiguous 
 q  11 c  37 cursor     15 candidate  14 cursor_id     15 candidate_id  14 idc   e is_contiguous 
 q  10 c  38 cursor     14 candidate  13 cursor_id     14 candidate_id  13 idc   d is_contiguous 
 q   9 c  39 cursor     13 candidate  12 cursor_id     13 candidate_id  12 idc   c is_contiguous 
 q   8 c  40 cursor     12 candidate  11 cursor_id     12 candidate_id  11 idc   b is_contiguous 
 q   7 c  41 cursor     11 candidate  10 cursor_id     11 candidate_id  10 idc   a is_contiguous 
 q   6 c  42 cursor     10 candidate   9 cursor_id     10 candidate_id   9 idc   9 is_contiguous 
 q   5 c  43 cursor      9 candidate   5 cursor_id      9 candidate_id   6 idc   6
 q   5 c  43 cursor      9 candidate   4 cursor_id      9 candidate_id   5 idc   5
 q   5 c  43 cursor      9 candidate   3 cursor_id      9 candidate_id   4 idc   4
 q   5 c  43 cursor      9 candidate   2 cursor_id      9 candidate_id   3 idc   3
 q   5 c  43 cursor      9 candidate   1 cursor_id      9 candidate_id   2 idc   2
 q   5 c  43 cursor      9 candidate   0 cursor_id      9 candidate_id   1 idc   1
 q   5 c  43 cursor      9 candidate   5 cursor_id      9 candidate_id   6 idc   6
 q   5 c  43 cursor      9 candidate   4 cursor_id      9 candidate_id   5 idc   5
 q   5 c  43 cursor      9 candidate   3 cursor_id      9 candidate_id   4 idc   4


    */

 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    int ctrl = 666 ;  // hexpatch

    test_are_contiguous(ctrl); 
    test_sort_contiguous(ctrl); 

    return 0 ; 
}
  
  
