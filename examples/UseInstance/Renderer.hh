#pragma once

#include "USEINSTANCE_API_EXPORT.hh"
#include <vector>
struct Buf ; 

struct USEINSTANCE_API Renderer
{
    unsigned vao;
    std::vector<Buf*> buffers ; 

    Renderer();

    void upload(Buf* buf);
    void destroy();
};


