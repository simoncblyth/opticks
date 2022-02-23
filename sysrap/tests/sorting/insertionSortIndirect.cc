// name=insertionSortIndirect ; gcc $name.cc -std=c++11 -lstdc++ -I$OPTICKS_PREFIX/include/SysRap -o /tmp/$name && /tmp/$name

/**
Exploring How To Sort CSG Leaf nodes into intersect Enter order for n-ary intersection
------------------------------------------------------------------------------------------

Problem characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~

* typically small number of leaf Enter for each ray, ie 1,2,3,4,5 
  (depending on layout of solid constituents)

* t_enter array will often be mostly sorted (or reversed) initially as leaf sub-ordering 
  usually follows geometrical positioning. Reversed as rays come from anywhere in any direction.

* dont actually want to sort an array of t_enter, want to sort the CSGNode* pointers of the leaves
  based on their intersect distances and do not want to keep repeating the intersects : 
  so collect array of t_enter and and indices array and do indirect sort  (like np.argsort)

* only a subset of all subs have Enter for each ray : requires mapping from enter index
  back to sub index OR planting a large token value  for non-Enter and doing some sweeping 
  before sorting

* currently using insertion sort small size probably means hard coded 
  sorting networks for : 1,2,3,4,5,6,7,8  would be better

  * BUT : needs to be indirect, and need mapping back to isub 


Available sort techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://en.wikipedia.org/wiki/Selection_sort

https://en.wikipedia.org/wiki/Sorting_network

https://en.wikipedia.org/wiki/Bitonic_sorter

https://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

https://stackoverflow.com/questions/32172144/fastest-way-to-sort-10-numbers-numbers-are-32-bit

https://www.toptal.com/developers/sorting-algorithms

.. selection sort is greatly outperformed on larger arrays by divide-and-conquer algorithms such as mergesort. 
However, insertion sort or selection sort are both typically faster for small arrays (i.e. fewer than 10–20 elements). 
A useful optimization in practice for the recursive algorithms is to switch to insertion sort or selection sort 
for "small enough" sublists.


Insertion Sort Algorithm 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To sort an array of size n in ascending order: 

1. Iterate from arr[1] to arr[n] over the array. 
2. Compare the current element (key) to its predecessor. 
3. If the key element is smaller than its predecessor, compare it to the
   elements before. Move the greater elements one position up to make space for
   the swapped element.

**/

#include "NP.hh"
#include <iomanip>


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

void insertionSortIndirect_0( int* idx, const float* enter, int count, int* aux )
{
    int i, j ; 

    for (i = 1; i < count ; i++) 
    {  
        int key = idx[i] ;    
        j = i - 1 ;   

        // descending j below i whilst find out of order  
        while( j >= 0 && enter[idx[j]] > enter[key] )     // need to use the idx here otherwise not seeing the shuffle
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

void insertionSortIndirect_1( int* idx, const float* enter, int count, int* aux )
{
    int i, j ; 

    for (i = 1; i < count ; i++) 
    {  
        int key = idx[i] ;    
        int akey = aux ? aux[i] : -1  ; 

        j = i - 1 ;   

        // descending j below i whilst find out of order  
        while( j >= 0 && enter[idx[j]] > enter[key] )     // need to use the idx here otherwise not seeing the shuffle
        {
            idx[j+1] = idx[j] ;  

            if(aux) aux[j+1] = aux[j] ; 

            // sliding values that are greater than the key one upwards
            // no need to "swap" as are holding the key out of the pack
            // ready to place it into the slot opened by the slide up   

            j = j - 1; 
        }   

        idx[j+1] = key ; 
        if(aux) aux[j+1] = akey ; 

        // if values below i are already ascending, then this 
        // puts the key back in the pack at the same place it came from  
    }   
}



void insertionSortIndirect_1_WRONG( int* idx, const float* enter, int count )
{
    int i, j ; 

    for (i = 1; i < count ; i++) 
    {  
        int key = idx[i] ;    
        j = i - 1 ;   

        // descending j below i whilst find out of order  
        while( j >= 0 && enter[j] > enter[i] )     // <--- WRONG : NEEDS TO BE enter[idx[j]] > enter[key] TO FEEL THE SHUFFLE
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

/**
THIS IS CORRECT BUT UNCLEAR COMPARED TO OTHER IMPS : MAYBE IT DOES MORE SHUFFLES WITH THE SWAPPING ?
**/

void insertionSortIndirect_2( int* idx, const float* enter, int count, int* aux )
{
    for (int i = 1; i < count ; i++)
    {
        for (int j = i; j > 0 && enter[idx[j]] < enter[idx[j-1]] ; j-- )
        {
            int swap = idx[j] ; 
            idx[j] = idx[j-1] ; 
            idx[j-1] = swap ; 

            if(aux)
            {
                int a_swap = aux[j] ; 
                aux[j] = aux[j-1] ; 
                aux[j-1] = a_swap ; 
            }
        }
    }
}



void insertionSortIndirect( int* idx, const float* enter, int count, int* aux, int imp )
{
    switch(imp)
    {
       case 0: insertionSortIndirect_0(idx, enter, count, aux);  break ; 
       case 1: insertionSortIndirect_1(idx, enter, count, aux);  break ; 
       case 2: insertionSortIndirect_2(idx, enter, count, aux);  break ; 
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


// TODO: mock the isub subsetting 
// to make a fuller test of CSG/csg_intersect_node.h



void test_insertionSortIndirect(const NP* e, int imp, const char* fold)
{
    const float* enter = e->cvalues<float>(); 
    int enter_count = e->shape[0]; 

    NP* i = NP::Make<int>(enter_count);
    i->fillIndexFlat();  
    int* idx = i->values<int>(); 
    int* aux = nullptr ; 

    insertionSortIndirect(idx, enter, enter_count, aux, imp ); 

    std::string name = U::FormName("i", imp, ".npy");     
    std::cout << "insertionSortIndirect saving to " << fold << "/" << name  << std::endl ; 
    i->save(fold, name.c_str() );  
}


void test_insertionSortIndirect(const char* fold)
{
    NP* e = NP::Load(fold, "e.npy"); 

    test_insertionSortIndirect(e, 0, fold);
    test_insertionSortIndirect(e, 1, fold);
    test_insertionSortIndirect(e, 2, fold);     
}



void dump( const int* idx, const float* enter, int count, const int* aux, const char* msg, const int* aux0 )
{
    int w = 30 ; 
    int v = 7 ; 

    std::cout << msg << " count " << count << std::endl ; 
    std::cout << std::setw(w) << " idx[i] "  ; 
    for(int i=0 ; i < count ; i++) std::cout << std::setw(v) << idx[i] << " " ; 
    std::cout << std::endl ; 

    if(aux)
    {
        std::cout << std::setw(w) << " aux[i] "  ; 
        for(int i=0 ; i < count ; i++) std::cout << std::setw(v) << aux[i] << " " ; 
        std::cout << std::endl ; 
    }
    if(aux0)
    {
        std::cout << std::setw(w) << " aux0[i] "  ; 
        for(int i=0 ; i < count ; i++) std::cout << std::setw(v) << aux0[i] << " " ; 
        std::cout << " (original aux0, unshuffled by sorting, maps enter index to value index ) " << std::endl ; 


        std::cout << std::setw(w) <<  " aux0[idx[i]] " ; 
        for(int i=0 ; i < count ; i++) std::cout << std::setw(v) << aux0[idx[i]] << " " ; 
        std::cout << " using shuffled idx order to get original value index in sort order " << std::endl ; 
    }


    std::cout <<  std::setw(w) << " enter[i] " ; 
    for(int i=0 ; i < count ; i++) std::cout << std::setw(v) << enter[i] << " " ; 
    std::cout << " (untouched by sorting) " << std::endl ; 

    std::cout << std::setw(w) <<  " enter[idx[i]] " ; 
    for(int i=0 ; i < count ; i++) std::cout << std::setw(v) << enter[idx[i]] << " " ; 
    std::cout << " using shuffled idx to get sorted enters " << std::endl ; 






}


void test_small()
{
    NP* e = NP::FromString<float>("10 110 -20 120 30 130 -10"); 
    float* enter = e->values<float>(); 
    int enter_count = e->shape[0]; 

    NP* i = NP::Make<int>(enter_count);
    i->fillIndexFlat();  
    int* idx = i->values<int>(); 

    int* aux = nullptr ; 
    int* aux0 = nullptr ; 

    dump(idx, enter, enter_count, aux, "before sort", aux0 ); 

    //insertionSort( enter, enter_count ); 

    int imp = 0 ; 
    insertionSortIndirect(idx, enter, enter_count, aux, imp ); 

    dump(idx, enter, enter_count, aux, "after sort", aux0 ); 
}




NP* make_idx(int n)
{
    NP* i = NP::Make<int>(n);
    i->fillIndexFlat();  
    //int* idx = i->values<int>(); 
    return i ; 
} 



void test_small_isub()
{
    NP* v = NP::FromString<float>("-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 10 110 20 120 30 130 43"); 
    float* vals = v->values<float>(); 
    int  vals_count = v->shape[0]; 

    
    float enter[32] ; 
    int   aux0[32] ;
    int   aux[32] ;
    int   idx[32] ;
    int enter_count = 0 ; 

    for(int i=0 ; i < vals_count ; i++)
    {
        float v = vals[i] ; 
        if( v > 0.f)
        {  
            enter[enter_count] = v ; 
            aux0[enter_count] = i ; // maps from enter index to original value index             
            aux[enter_count] = i ;  // ATTEMPTING 2 LEVELS OF INDIRECTION : NOT GOING TO WORK WITH SORT  
            idx[enter_count] = enter_count ;   
            // ONE LEVEL INDIRECT IS OK
            enter_count += 1 ; 
            assert( enter_count < 32 ); 
        }
    }

    
   /*
    NP* e = NP::Make<float>(enter_count);
    e->read2(&enter[0]) ; 

    NP* i = NP::Make<int>(enter_count);
    i->read2(&[0]) ; 

   */

    int imp = 1 ; 

    dump(idx, enter, enter_count, aux, "before sort", aux0 ); 

    insertionSortIndirect(idx, enter, enter_count, aux, imp ); 

    dump(idx, enter, enter_count, aux, "after sort", aux0  ); 



}




int main(int argc, char** argv)
{
    //test_insertionSortIndirect("/tmp"); 
    //test_small();  
    test_small_isub();  

    return 0 ; 
}
