#include "GOpticalSurface.hh"
#include "OPTICKS_LOG.hh"

void test_ctor(char finish_)
{
    const char* name = "dummy" ; 
    const char* type = "" ; 
    const char* model = "" ; 
    const char* value = "0" ; 

    char finish[2] = {finish_, 0} ;   // null termination 

    GOpticalSurface os(name, type, model, (const char*)finish, value ); 

    unsigned u_finish = os.getFinishInt() ; 
    LOG(info) << " u_finish " << u_finish <<  " os.description " << os.description()  ; 

}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    char finish[] = "012345" ; 

    for(unsigned i=0 ; i < sizeof(finish) - 1 ; i++) test_ctor(finish[i]) ; 



    return 0 ; 
}
