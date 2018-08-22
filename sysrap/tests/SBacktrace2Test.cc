// TEST=SBacktrace2Test om-t 

#include <string>
#include <cxxabi.h>

#include "OPTICKS_LOG.hh"

std::string macOS = R"(
4   libG4processes.dylib                0x00000001090baee5 _ZN12G4VEmProcess36PostStepGetPhysicalInteractionLengthERK7G4TrackdP16G4ForceCondition + 661
5   libG4tracking.dylib                 0x00000001088ffff0 _ZN10G4VProcess12PostStepGPILERK7G4TrackdP16G4ForceCondition + 80
6   libG4tracking.dylib                 0x00000001088ffa1a _ZN17G4SteppingManager24DefinePhysicalStepLengthEv + 298
7   libG4tracking.dylib                 0x00000001088fcc3a _ZN17G4SteppingManager8SteppingEv + 394
8   libG4tracking.dylib                 0x000000010891386f _ZN17G4TrackingManager15ProcessOneTrackEP7G4Track + 1679
9   libG4event.dylib                    0x00000001087da71a _ZN14G4EventManager12DoProcessingEP7G4Event + 3306
10  libG4event.dylib                    0x00000001087dbc2f _ZN14G4EventManager15ProcessOneEventEP7G4Event + 47
11  libG4run.dylib                      0x00000001086e79f5 _ZN12G4RunManager15ProcessOneEventEi + 69
12  libG4run.dylib                      0x00000001086e7825 _ZN12G4RunManager11DoEventLoopEiPKci + 101
13  libG4run.dylib                      0x00000001086e5ce1 _ZN12G4RunManager6BeamOnEiPKci + 193
14  libCFG4.dylib                       0x0000000106a63df9 _ZN3CG49propagateEv + 1689
15  libOKG4.dylib                       0x00000001000e22b6 _ZN7OKG4Mgr10propagate_Ev + 182
16  libOKG4.dylib                       0x00000001000e1ec6 _ZN7OKG4Mgr9propagateEv + 470
17  OKG4Test                            0x0000000100014c89 main + 489
18  libdyld.dylib                       0x00007fff6bd8b015 start + 1
19  ???                                 0x0000000000000005 0x0 + 5
)" ; 


struct frame 
{
    frame( char* line_ ) 
        :
        line(line_),
        name(NULL),
        offset(NULL),
        func(NULL)
    {
        parse(); 
    } 

    ~frame()
    {
       free(func);  
    }

    void parse()
    {
        for( char *p = line ; *p ; ++p )
        {   
            if (( *p == '_' ) && ( *(p-1) == ' ' )) name = p-1;
            else if ( *p == '+' ) offset = p-1;
        }   

        if( name && offset && ( name < offset ))
        {   
            *name++ = '\0';    // plant terminator into line
            *offset++ = '\0';  // plant terminator into name  
            func = demangle(); 
        }  
    }

    char* demangle() // demangling fails for non C++ symbols
    {
        int status;
        char* ret = abi::__cxa_demangle( name, NULL, NULL, &status );
        return status == 0 ? ret : NULL ; 
    }

    void dump()
    {
         if( func )
         {
             std::cout 
                   << std::setw(30) << line  
                   << " "
                   << std::setw(100) << std::left << func
                   << " "
                   << std::setw(10) << std::left << offset 
                   << std::endl 
                   ;
         }
         else
         {
             std::cout  << line << std::endl ; 

         }
    }

    char* line ; 
    char* name ; 
    char* offset ;
 
    char* func ;    // only func is "owned"

};



int main(int  argc, char** argv )
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << std::endl << macOS ; 

    const char* lines = macOS.c_str(); 
    char delim = '\n' ; 

    std::istringstream f(lines);
    std::string line ;

    while (getline(f, line, delim))
    {   
        if(line.empty()) continue ; 

        frame f((char*)line.c_str());
        f.dump(); 
    }   

    return 0 ; 
}

