#pragma once

#include "QUDARAP_API_EXPORT.hh"
#include "vector_types.h"

struct QUDARAP_API QPoly
{
    QPoly(); 
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );
    void demo(); 
    void tmpl_demo(); 

};


