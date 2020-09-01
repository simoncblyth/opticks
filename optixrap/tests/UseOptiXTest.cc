/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <optix_world.h>

// from SDK/sutil/sutil.h

struct APIError
{   
    APIError( RTresult c, const std::string& f, int l ) 
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

// Error check/report helper for users of the C API 
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw APIError( code, __FILE__, __LINE__ );           \
  } while(0)





struct Args
{
    bool ctx ; 
    bool cmdline ; 
    char* cvd ; 
    bool names ; 
    bool bus ; 
    bool uniqnames ; 
    bool uniqrec ; 
    bool ordinals ; 
    bool num ; 
    bool verbose ; 

    Args( int argc, char** argv )
        :
        ctx(false),
        cmdline(false),
        cvd(NULL),
        names(false),
        bus(false),
        uniqnames(false),
        uniqrec(false),
        ordinals(false),
        num(false),
        verbose(false)
    {
        char buf[256] ; 
        for(int i=1 ; i < argc ; i++)
        {
            if(i < argc - 1 && strcmp(argv[i], "--cvd") == 0) 
            {
                snprintf(buf, 256, "CUDA_VISIBLE_DEVICES=%s", argv[i+1]) ; 
                cvd = strdup(buf) ; 
            }
            else if( i < argc && strcmp(argv[i], "--ctx" ) == 0)
            {
                ctx = true ;    
            } 
            else if( i < argc && strcmp(argv[i], "--cmdline" ) == 0)
            {
                cmdline = true ;    
            } 
            else if( i < argc && strcmp(argv[i], "--names" ) == 0)
            {
                names  = true ; 
            }
            else if( i < argc && strcmp(argv[i], "--bus" ) == 0)
            {
                bus  = true ; 
            }
            else if( i < argc && strcmp(argv[i], "--ordinals" ) == 0)
            {
                ordinals  = true ; 
            }
            else if( i < argc && strcmp(argv[i], "--uniqnames" ) == 0)
            {
                uniqnames  = true ; 
            }
            else if( i < argc && strcmp(argv[i], "--uniqrec" ) == 0)
            {
                uniqrec  = true ; 
            }
            else if( i < argc && strcmp(argv[i], "--num" ) == 0)
            {
                num  = true ; 
            }
            else if( i < argc && strcmp(argv[i], "--verbose" ) == 0)
            {
                verbose = true ; 
            }
        }

        if(cmdline)
        {
            std::cout << "UseOptiX" ; 
            for(int i=1 ; i < argc ; i++) std::cout << " " << argv[i] ; 
            std::cout << std::endl ; 
        }
    }  
    bool quiet() const 
    {
        return verbose ? false : names || uniqnames || uniqrec || num || ordinals || cmdline || bus ; 
    }

         


};


struct Devices
{
    enum { MAX_DEVICES = 8 };

    const Args& args ; 
    unsigned num_devices;
    unsigned version;

    std::vector<std::string> names ; 
    std::vector<std::string> uniqs ; 
    std::vector<std::string> bus ; 
    std::vector<int> ordinals ; 
    std::vector<std::string> uniqified ; 
    std::vector<std::string> uniqrec ; 

    Devices( const Args& args_ )  : args(args_)
    {
        init();
        uniqify(); 
        uniqrecify(); 
    }

    void init()
    {
        // extracts from /Developer/OptiX/SDK/optixDeviceQuery/optixDeviceQuery.cpp
        RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
        RT_CHECK_ERROR(rtGetVersion(&version));
        assert( num_devices <= MAX_DEVICES ); 

        if(!args.quiet())
        {
            printf("OptiX version %d major.minor.micro %d.%d.%d   Number of devices = %d \n\n", 
                 version, version / 10000, (version % 10000) / 100, version % 100, num_devices );
        }

        for(unsigned i = 0; i < num_devices; ++i) 
        {
            char name[256];
            char busid[256];
            int computeCaps[2];
            int compat[MAX_DEVICES+1];
            int ordinal ; 
            RTsize total_mem;

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal));

#if OPTIX_VERSION_MAJOR >= 6
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(busid), busid));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES, sizeof(compat), &compat));
#endif


            ordinals.push_back(ordinal);

            char* p = name ; 
            while (*p)
            {
                if (*p == ' ') *p = '_';
                p++ ; 
            } 
            names.push_back(name); 
            if(std::find(uniqs.begin(),uniqs.end(), name) == uniqs.end())  uniqs.push_back(name);

            bus.push_back(busid);   

            if(!args.quiet())
            {
                printf(" Device %d: %30s  ordinal:%d  ", i, name, ordinal  );
#if OPTIX_VERSION_MAJOR >= 6
                printf(" busid: %15s compat[0]:%d ", busid, compat[0] );
#endif
                printf(" Compute Support: %d %d ", computeCaps[0], computeCaps[1]);
                printf(" Total Memory: %llu bytes \n", (unsigned long long)total_mem);
            }
        } 
    } 


    void uniqify()
    {
        assert( names.size() == ordinals.size() ); 
        if( names.size() == uniqs.size() )
        {
            if(!args.quiet()) std::cout << "all GPU names are unique, nothing to do " << std::endl ; 
            for(unsigned i = 0 ; i < names.size() ; i++) 
            {
                uniqified.push_back( names[i] ); 
            } 
        }    
        else
        {
            for(unsigned i = 0 ; i < names.size() ; i++)
            {
                std::stringstream ss ;  
                ss << names[i] << "-" << ordinals[i] ;  
                std::string uname = ss.str();
                uniqified.push_back(uname); 
            }
        }
    }


    void uniqrecify()
    {
        assert( names.size() == ordinals.size() ); 
        assert( names.size() == uniqified.size() ); 
           
        for(unsigned i = 0 ; i < names.size() ; i++)
        {
            std::stringstream ss ;  
            ss << uniqified[i] << "/" << ordinals[i] ;  
            std::string urec = ss.str();
            uniqrec.push_back(urec); 
        } 
    }


    void dump_num() const { std::cout << num_devices << std::endl ; }
    void dump_vec(const std::vector<std::string>& vec) const
    {
        for(unsigned i = 0 ; i < vec.size() ; i++) std::cout << vec[i] << std::endl ;  
    }
    void dump_vec(const std::vector<int>& vec) const
    {
        for(unsigned i = 0 ; i < vec.size() ; i++) std::cout << vec[i] << std::endl ;  
    }


    void dump_names()    const { dump_vec(names) ; } 
    void dump_ordinals() const { dump_vec(ordinals) ; } 
    void dump_uniqnames() const { dump_vec(uniqified) ; } 
    void dump_uniqrec()  const { dump_vec(uniqrec) ; } 
    void dump_bus() const { dump_vec(bus) ; } 

};




int main(int argc, char** argv)
{
    Args args(argc, argv); 
    
    if(args.cvd)
    {
        if(!args.quiet()) printf("setting CUDA_VISIBLE_DEVICES envvar internally : %s\n", args.cvd );
        putenv(args.cvd);  
    }

    Devices devs(args) ; 

    if( args.num) devs.dump_num(); 
    if( args.names ) devs.dump_names(); 
    if( args.ordinals ) devs.dump_ordinals(); 
    if( args.uniqnames ) devs.dump_uniqnames(); 
    if( args.uniqrec ) devs.dump_uniqrec() ; 
    if( args.bus ) devs.dump_bus() ; 

    if(!args.ctx) return 0 ; 

    RTcontext context = 0;

    //RTprogram program;
    RTbuffer  buffer;
    RTvariable variable ; 

    int width = 1024u ; 
    int height = 768u ; 
   
    std::cout << std::endl << std::endl ; 
    std::cout << "( creating context " << std::endl ; 
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    std::cout << ") creating context " << std::endl ; 
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, width, height ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "variable", &variable ) );

    RTformat format = RT_FORMAT_FLOAT4 ; 
    size_t size = 0 ; 
    RT_CHECK_ERROR( rtuGetSizeForRTformat( format, &size) ); 

    std::cout << " RT_FORMAT_FLOAT4 size " << size << std::endl ; 
    assert( size == sizeof(float)*4 ) ; 

    return 0 ; 
}

