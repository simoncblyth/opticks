#pragma once

#include "scuda.h"
#include "stran.h"

template <typename T>
struct sframe
{
    glm::tvec4<T> ce ;  
    Tran<T>*  geotran ;    
}; 




