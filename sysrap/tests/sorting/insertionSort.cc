// name=insertionSort ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name
// https://www.geeksforgeeks.org/insertion-sort/

#include <cmath>
#include <cstdio>
#include <cassert>

/** 

Move elements of a[0..i-1], that are
greater than key, to one position ahead
of their current position 
**/

void insertionSort(int* a, int n)
{
	int i, key, j;
	for (i = 1; i < n; i++) 
    {
		key = a[i];
		j = i - 1;

		while (j >= 0 && a[j] > key) 
        {
			a[j + 1] = a[j];
			j = j - 1;
		}
		a[j + 1] = key;
	}
}

/**
insertionSortIndirect
----------------------

Expects *idx* to start as 0,1,2,3,...,n-1

This indices are shuffled in a way that would make
the values of *a* to be in ascending order 
without touching a. 

https://algs4.cs.princeton.edu/21elementary/Insertion.java.html

**/

void insertionSortIndirect(int* idx, const int* a, int n)
{
	for (int i = 1; i < n; i++) 
    {
        for(int j = i ; j > 0 && a[idx[j]] < a[idx[j-1]] ; j-- )
        {
             int swap = idx[j] ; 
             idx[j] = idx[j-1]; 
             idx[j-1] = swap ; 
        } 
	}
}

/**
insertionSortIndirectSentinel
---------------------------------

The sorting stops on reaching a with the sentinel value.

This assumes that sweepSentinel has been invoked previously
to prime the indices. 

**/

void insertionSortIndirectSentinel(int* idx, const int* a, int n, int sentinel)
{
	for (int i = 1; i < n; i++) 
    {
        if( a[idx[i]] == sentinel ) return ; 

        for(int j = i ; j > 0 && a[idx[j]] < a[idx[j-1]] ; j-- )
        {
             int swap = idx[j] ; 
             idx[j] = idx[j-1]; 
             idx[j-1] = swap ; 
        } 
	}
}



/**
sweepSentinel
---------------

Indices are shuffled such to indirectly move sentinel values 
into far right "high value" slots.
This is done to prevent having to move them via the sort. 
 
**/

void sweepSentinel(int* idx, const int* a, int n, int sentinel)
{
    int j = n-1 ; 
	for (int i = 0; i < n; i++) 
    {
        if(sentinel == a[idx[i]] && i < j)
        {
            int swap = idx[j] ; 
            idx[j] = idx[i] ; 
            idx[i] = swap ;  
            j-- ;  
        }  
    }
}




void printArray(int* a, int n)
{
	int i;
	for (i = 0; i < n; i++) printf("%d ", a[i]);
	printf("\n");
}

void printArrayIndirect(int* idx, int* a, int n)
{
	int i;
	for (i = 0; i < n; i++) printf("%5d ", idx[i]);
	printf("\n");
	for (i = 0; i < n; i++) printf("%5d ", a[idx[i]]);
	printf("\n");

}



int* create_sample(int n, int sentinel )
{
    assert( n == 5 ); 
    int* a = new int[n] ; 
    a[0] = 12 ;     
    a[1] = 11 ;     
    a[2] = 13 ;     
    a[3] = 5 ;     
    a[4] = 6 ;

    if(sentinel > 0)
    {
       a[3] = sentinel ; 
    }
     
    return a ;  
}

int* create_indices(int n)
{
    int* idx = new int[n] ; 
    for(int i=0 ; i < n ; i++)  idx[i] = i ; 
    return idx ; 
}

void test_insertionSort()
{
    int n = 5 ; 
    int* a = create_sample(n, 0); 
	insertionSort(a, n);
	printArray(a, n);
}

void test_insertionSortIndirect()
{
    int n = 5 ; 
    printf("test_insertionSortIndirect %d \n", n); 

    int* a = create_sample(n, 0); 
    int* idx = create_indices(n); 

	printArrayIndirect(idx, a, n);

	insertionSortIndirect(idx, a, n);

	printArrayIndirect(idx, a, n);
}


void test_sweepSentinel()
{
    int n = 5 ; 
    int sentinel = 10000 ; 
    int* a = create_sample(n, sentinel); 
    int* idx = create_indices(n); 

	sweepSentinel(idx, a, n, sentinel);
	printArrayIndirect(idx, a, n);
}


void test_insertionSortIndirectSentinel()
{
    int n = 5 ; 
    int sentinel = 100000 ; 

    printf("test_insertionSortIndirectSentinel %d \n", n); 

    int* a = create_sample(n, sentinel); 
    int* idx = create_indices(n); 

	printArrayIndirect(idx, a, n);

	sweepSentinel(idx, a, n, sentinel);

	insertionSortIndirectSentinel(idx, a, n, sentinel);

	printArrayIndirect(idx, a, n);
}






int main()
{
    //test_insertionSort(); 
    //test_insertionSortIndirect(); 
    //test_sweepSentinel(); 
    test_insertionSortIndirectSentinel(); 

	return 0;
}


