// name=insertionSortIndirect ; gcc $name.cc -std=c++11 -lstdc++ -I$OPTICKS_PREFIX/include/SysRap -o /tmp/$name && /tmp/$name
#include "NP.hh"

#include <iomanip>



/**

Algorithm 
To sort an array of size n in ascending order: 

1. Iterate from arr[1] to arr[n] over the array. 
2. Compare the current element (key) to its predecessor. 
3. If the key element is smaller than its predecessor, compare it to the
   elements before. Move the greater elements one position up to make space for
   the swapped element.

**/

void insertionSortIndirect_0( int* idx, const float* enter, int count )
{
    int i, j ; 

    for (i = 1; i < count ; i++) 
    {  
        int key = idx[i] ;    
        j = i - 1 ;   

        // descending j below i whilst find out of order  
        while( j >= 0 && enter[idx[j]] > enter[key] )  
        {
            idx[j+1] = idx[j] ;  

            // sliding values that are greater than the key one upwards
            // no need to "swap" as are holding the key out of the pack
            // ready to place it into the slot opened by the slide up   

            j = j - 1; 
        }   

        idx[j+1] = key ; 

        // if values below i are already ascending, then this 
        // puts the key back in the pack at the same place it came from  
    }   
}


void insertionSortIndirect_1( int* idx, const float* enter, int count )
{
    for (int i = 1; i < count ; i++)
    {
        for (int j = i; j > 0 && enter[idx[j]] < enter[idx[j-1]] ; j-- )
        {
            int swap = idx[j] ; 
            idx[j] = idx[j-1] ; 
            idx[j-1] = swap ; 
        }
    }
}

void insertionSortIndirect( int* idx, const float* enter, int count )
{
    insertionSortIndirect_0(idx, enter, count); 
}


void insertionSort(float* enter, int n)
{
    /* Move elements of arr[0..i-1], that are
       greater than key, to one position ahead
       of their current position */

    int i, j;

    for (i = 1; i < n; i++) 
    {
        float key = enter[i];   // move key thru the array 
        j = i - 1;

        // j is less than i, so want the enter[j] values to be less than key
        while (j >= 0 && enter[j] > key) 
        {
            enter[j + 1] = enter[j];
            j = j - 1;    // keep decrementing j down to zero 
        }
        enter[j + 1] = key;
    }
}

unsigned compare( const int* idx0, const int* idx1, unsigned count )
{
    unsigned mm = 0 ; 
    for(int i=0 ; i < count ; i++)
    {
       int i0 = idx0[i] ; 
       int i1 = idx1[i] ; 
       std::cout 
           << " i0 " << std::setw(5) << i0 
           << " i1 " << std::setw(5) << i1
           << " mm " << std::setw(5) << mm
           << std::endl
           ;

       if(i0 != i1) mm +=1 ; 
    }
    return mm ; 
}


void test_load_save(int argc, char** argv)
{
    const char* dir_ = argc > 1 ? argv[1] : "/tmp" ; 

    NP* e = NP::Load(dir_, "e.npy"); 
    float* enter = e->values<float>(); 
    int enter_count = e->shape[0]; 

    NP* i0 = NP::Make<int>(enter_count);
    i0->fillIndexFlat();  
    int* idx0 = i0->values<int>(); 

    NP* i1 = NP::Make<int>(enter_count);
    i1->fillIndexFlat();  
    int* idx1 = i1->values<int>(); 



    // TODO: mock the isub subsetting 
    // to make a fuller test of CSG/csg_intersect_node.h


    insertionSortIndirect_0(idx0, enter, enter_count); 
    insertionSortIndirect_1(idx1, enter, enter_count); 

    unsigned mismatch = compare( idx0, idx1, enter_count );  
    assert( mismatch == 0 ); 

    std::cout << "insertionSortIndirect saving to " << dir_ << std::endl ; 

    i0->save(dir_, "i0.npy");  
    i1->save(dir_, "i1.npy");  
}

void dump( const int* idx, const float* enter, int count, const char* msg )
{
    std::cout << msg << " count " << count << std::endl ; 
    std::cout << std::setw(20) << " idx[i]"  ; 
    for(int i=0 ; i < count ; i++) std::cout << idx[i] << " " ; 
    std::cout << std::endl ; 

    std::cout <<  std::setw(20) << " enter[i]" ; 
    for(int i=0 ; i < count ; i++) std::cout << enter[i] << " " ; 
    std::cout << std::endl ; 

    std::cout << std::setw(20) <<  " enter[idx[i]]" ; 
    for(int i=0 ; i < count ; i++) std::cout << enter[idx[i]] << " " ; 
    std::cout << std::endl ; 
}

void test_small()
{
    NP* e = NP::FromString<float>("10 110 20 120 30 130 -10"); 
    float* enter = e->values<float>(); 
    int enter_count = e->shape[0]; 

    NP* i = NP::Make<int>(enter_count);
    i->fillIndexFlat();  
    int* idx = i->values<int>(); 


    dump(idx, enter, enter_count, "before sort" ); 

    //insertionSort( enter, enter_count ); 

    insertionSortIndirect(idx, enter, enter_count); 



    dump(idx, enter, enter_count, "after sort" ); 
}




int main(int argc, char** argv)
{
    test_load_save(argc, argv); 
    //test_small();  
    return 0 ; 
}
