#include "STimes.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{  
    PLOG_(argc, argv);

    STimes st ; 
    st.validate = 0.010 ;
    st.compile  = 0.020 ;
    st.prelaunch = 0.030 ;
    st.launch = 0.040 ;
     
    LOG(info) << st.brief("st:") ; 

    return 0 ; 
}


