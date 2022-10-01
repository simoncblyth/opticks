#pragma once

#define DEMO_API  __attribute__ ((visibility ("default")))

struct DEMO_API DEMO
{
    static void Dump(); 
}; 
