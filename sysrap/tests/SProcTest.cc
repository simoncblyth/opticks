#include <cassert>
#include <iomanip>
#include "SProc.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    int M = 1000000 ; 
    unsigned nleak = 25 ;    

    char** leaks = new char*[nleak] ; 

    float vmb0 = SProc::VirtualMemoryUsageMB();

    LOG(info) << " vmb0 " << vmb0 ; 

    for(unsigned i=0 ; i < nleak ; i++)
    {
        leaks[i] = new char[100*M] ; 

        float dvmb = SProc::VirtualMemoryUsageMB() - vmb0 ;

        assert(leaks[i]); 

        LOG(info) 
              << std::setw(10) << i 
              << " vm " 
              << std::setw(10) << dvmb 
              ;

        //delete [] leaks[i] ; 
    } 
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
