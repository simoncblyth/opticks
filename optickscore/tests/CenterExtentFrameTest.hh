#pragma once

#include <vector>
#include <map>
template <typename T> struct CenterExtentFrame ; 

template<typename T>
struct CenterExtentFrameTest 
{
    CenterExtentFrameTest( const CenterExtentFrame<T>& cef ); 

    const CenterExtentFrame<T>& cef ; 
    const glm::tvec4<T>& ce ; 

    std::vector<std::string> modes ;  
    std::vector<std::string> pref_modes ;  

    std::map<std::string, glm::tmat4x4<T>> _model2world ; 
    std::map<std::string, glm::tmat4x4<T>> _world2model ; 

    void check(const std::vector<glm::vec4>& world, const std::vector<std::string>& label, const char* title, const char*  w2m_mode, const char* m2w_mode ); 
    void check(char m='P');  
    void dump(char m='P');  
};


