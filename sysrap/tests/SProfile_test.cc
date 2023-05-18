#include <chrono>
#include <thread>

#include "SProfile_test.h"
#include "SProfile.h"

template<>
std::vector<SProfile<4>> SProfile<4>::RECORD = {}  ; 


SProfile_test::SProfile_test()
    :
    d(std::atoi(getenv("DELAY"))),  
    prof(new SProfile<4>) 
{
    std::chrono::microseconds delay(d);

    for(int i=0 ; i < 10 ; i++)  // eg over events
    {
        for(int j=0 ; j < 100 ; j++)  //eg over photons on the event 
        {
            *prof = {} ; 
            prof->idx = i*100+j ; 

            for(int k=0 ; k < 4 ; k++)  //eg over a few code sites to monitor 
            {
                prof->stamp(k) ; 
                std::this_thread::sleep_for(delay); 
            }
            prof->add(); 
        }
        std::string reldir = std::to_string(i) ; 
        SProfile<4>::Save("$FOLD", reldir.c_str() ); 
        SProfile<4>::Clear(); 
    }

}
  


int main()
{
    SProfile_test spt ; 
    return 0 ; 
}


