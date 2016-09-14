#pragma once
#include <string>
#include <vector>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksActionControl {
    public:
        enum {
                GS_LOADED      = 0x1 << 1,
                GS_FABRICATED  = 0x1 << 2,
                GS_TRANSLATED  = 0x1 << 3,
                GS_TORCH       = 0x1 << 4,
                GS_LEGACY      = 0x1 << 5
             };  
    public:
        static const char* GS_LOADED_  ; 
        static const char* GS_FABRICATED_ ; 
        static const char* GS_TRANSLATED_ ; 
        static const char* GS_TORCH_ ; 
        static const char* GS_LEGACY_ ; 
    public:
        static std::string Description(unsigned long long ctrl);
        static unsigned long long Parse(const char* ctrl, char delim=',');
        static unsigned long long ParseTag(const char* ctrl);
        static bool isSet(unsigned long long ctrl, const char* mask);
        static std::vector<const char*> Tags();
    public:
        OpticksActionControl(unsigned long long* ctrl); 
        void add(const char* mask);
        bool isSet(const char* mask) const;
        bool operator()(const char* mask) const;
        std::string description(const char* msg="OpticksActionControl::description") const;
    private:
         unsigned long long* m_ctrl ; 

};

#include "OKCORE_TAIL.hh"


