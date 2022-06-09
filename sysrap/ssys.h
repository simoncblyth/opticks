#pragma once
/**
ssys.h
========

**/

#include <string>
#include <sstream>

struct ssys
{
    static std::string popen(const char* cmd, bool chomp=true, int* rc=nullptr);      
}; 


inline std::string ssys::popen(const char* cmd, bool chomp, int* rc)
{
    std::stringstream ss ; 
    FILE *fp = ::popen(cmd, "r");
    char line[512];    
    while (fgets(line, sizeof(line), fp) != NULL) 
    {   
       if(chomp) line[strcspn(line, "\n")] = 0;
       ss << line ;   
    }   

    int retcode=0 ; 
    int st = pclose(fp);
    if(WIFEXITED(st)) retcode=WEXITSTATUS(st);

    if(rc) *rc = retcode ; 

    std::string s = ss.str(); 
    return s ; 
}



