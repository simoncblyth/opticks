#include "NPFold.h"


struct sleak
{
    const NP* run ; 
    NP* runprof ;

    sleak();  
    std::string desc() const; 
    NPFold* serialize() const ; 
    void    import( const NPFold* fold ); 
    void save(const char* dir) const ; 
    static sleak* Load(const char* dir) ; 
};

inline sleak::sleak()
    :
    run(nullptr),
    runprof(nullptr)
{
}

inline std::string sleak::desc() const 
{
    std::stringstream ss ; 
    ss << "sleak::desc" << std::endl 
       << " run :" << ( run ? run->sstr() : "no-run" ) << std::endl 
       << " runprof :" << ( runprof ? runprof->sstr() : "no-runprof" ) << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}  


inline NPFold* sleak::serialize() const 
{
    NPFold* smry = new NPFold ;  
    if(run) smry->add("run", run ) ; 
    if(runprof) smry->add("runprof", runprof ) ; 
    return smry ; 
}
inline void sleak::import(const NPFold* smry) 
{
    run = smry->get("run")->copy() ; 
    runprof = smry->get("runprof")->copy() ; 
}
inline void sleak::save(const char* dir) const 
{
    NPFold* smry = serialize(); 
    smry->save_verbose(dir); 
}
inline sleak* sleak::Load(const char* dir) // static
{
    NPFold* smry = NPFold::Load(dir) ; 
    sleak* leak = new sleak ; 
    leak->import(smry) ; 
    return leak ; 
}





struct sleak_Creator
{
    bool VERBOSE ; 
    const char* dirp ; 
    const NPFold* fold ; 
    bool fold_valid ; 
    const NP* run ; 
    sleak* leak ; 

    sleak_Creator( const char* dirp_ ); 
    std::string desc() const; 
    
}; 


inline sleak_Creator::sleak_Creator( const char* dirp_ )
    :
    VERBOSE(getenv("sleak_Creator__VERBOSE") != nullptr),
    dirp(dirp_ ? strdup(dirp_) : nullptr),
    fold(NPFold::LoadNoData(dirp)),
    fold_valid(NPFold::IsValid(fold)),
    run(fold_valid ? fold->get("run") : nullptr),
    leak(new sleak)
{
    leak->run = run->copy() ;  //  HUH: if dont copy get SEGV om saving (presumably due to NoData)
    leak->runprof = leak->run ? run->makeMetaKVProfileArray("Index") : nullptr ; 
}

inline std::string sleak_Creator::desc() const 
{
    std::stringstream ss ; 
    ss << "sleak_Creator::desc" << std::endl 
       << " dirp " << ( dirp ? dirp : "-" ) << std::endl 
       << " fold " << ( fold ? "YES" : "NO " ) << std::endl 
       << " fold_valid " << ( fold_valid ? "YES" : "NO " ) << std::endl 
       << " run :" << ( run ? run->sstr() : "no-run" ) << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}  

int main(int argc, char** argv)
{
    char* argv0 = argv[0] ; 
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;   
    if(dirp == nullptr) return 0 ; 
    sleak_Creator creator(dirp); 
    std::cout << creator.desc() ; 
    if(!creator.fold_valid) return 1 ; 

    const NP* run = creator.run ; 
    std::cout << " run.desc " << run->desc() << std::endl ; 

    sleak* leak = creator.leak ; 

    //std::cout << "run->meta" << std::endl << run->meta << std::endl ; 
    std::cout << "leak->desc" << std::endl << leak->desc() << std::endl ; 
    std::cout << " saving ...  " << std::endl ; 
    leak->save("$SLEAK_FOLD"); 


    return 0 ; 
}



