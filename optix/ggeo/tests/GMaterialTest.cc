
#include "GMaterial.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    GMaterial* mat = new GMaterial("test", 0);
    mat->Summary(); 



    return 0 ;
}

