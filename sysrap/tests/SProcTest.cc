// TEST=SProcTest om-t

#include <cassert>
#include <iomanip>
#include "SProc.hh"
#include "OPTICKS_LOG.hh"


void test_leaking(int argc, char** argv)
{
    typedef unsigned long long ULL ; 

    //ULL MB = 1000000 ; 
    ULL MB = 1 << 20 ;   // 1048576  

    ULL MB128 = 1 << 20 << 7 ; 

    int nleak = argc > 1 ? atoi(argv[1]) : 25 ;   // reduced default from 100, to prevent std::bad_alloc fail at ~5GB on lxslc702

    char** leaks = new char*[nleak] ; 

    float vmb0 = SProc::VirtualMemoryUsageMB();

    LOG(info) 
        << " nleak " << nleak
        << " vmb0 " << vmb0 
        ; 

    ULL increment = MB128 ;  

    ULL total = 0ull ; 

    for(int i=0 ; i < nleak ; i++)
    {
        leaks[i] = new char[increment] ;

        total += increment ;    

        float dvmb = SProc::VirtualMemoryUsageMB() - vmb0 ;

        float x_dvmb = float(total/MB) ;  

        assert(leaks[i]); 

        LOG(info) 
              << std::setw(10) << i 
              << " vm " 
              << std::setw(10) << dvmb 
              << " x_vm " 
              << std::setw(10) << x_dvmb 
              << " vm/x_vm " 
              << std::setw(10) << std::fixed << std::setprecision(4) << dvmb/x_dvmb 
 
              ;

        //delete [] leaks[i] ; 
    } 
}



void test_ExecutablePath(int argc, char** argv)
{
    const char* p = SProc::ExecutablePath();
    std::cout << "argv[0]:                 " << argv[0] << std::endl ;  
    std::cout << "SProc::ExecutablePath(): " << p << std::endl ;  

    const char* n = SProc::ExecutableName();
    std::cout << "SProc::ExecutableName(): " << n << std::endl ;  

    const char* l4 = strdup( n + strlen(n) - 4 ); 
    std::cout << " l4 [" << l4 << "]" << std::endl ; 
    assert( strlen(n) > 4 && strncmp( n + strlen(n) - 4 ,  "Test", 4) == 0 );  
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_leaking(argc, argv); 
    test_ExecutablePath(argc, argv); 

    return 0 ; 
}

/*
When skipping the delete the VM grows linearly at close to expected 100 MB for each loop::

    simon:sysrap blyth$ SProcTest 
    2016-09-14 20:31:57.967 INFO  [310051] [main@25]          0 vm        100
    2016-09-14 20:31:57.968 INFO  [310051] [main@25]          1 vm        209
    2016-09-14 20:31:57.968 INFO  [310051] [main@25]          2 vm        309
    ...
    2016-09-14 20:31:57.970 INFO  [310051] [main@25]         97 vm       9810
    2016-09-14 20:31:57.970 INFO  [310051] [main@25]         98 vm       9910
    2016-09-14 20:31:57.970 INFO  [310051] [main@25]         99 vm      10010


Bad alloc on lxslc702.ihep.ac.cn when get to 50th, ie 5000MB::

    2019-04-28 19:14:09.130 INFO  [17056] [main@27]         47 vm          0
    2019-04-28 19:14:09.130 INFO  [17056] [main@27]         48 vm          0
    2019-04-28 19:14:09.130 INFO  [17056] [main@27]         49 vm          0
    terminate called after throwing an instance of 'std::bad_alloc'
      what():  std::bad_alloc
    Aborted (core dumped)




*/
