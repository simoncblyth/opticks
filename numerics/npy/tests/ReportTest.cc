#include "Report.hpp"



void test_save()
{
    std::vector<std::string> lines ; 
    lines.push_back("hello");
    lines.push_back("world");

    Report r ; 
    r.add(lines);
    r.save("/tmp", "TestReport.txt");
    r.save("$IDPATH/tmp", Report::name("typ","tag").c_str() );
}

int main()
{
    const char* dir = "$LOCAL_BASE/env/opticks/rainbow/mdtorch/5/20151226_154520" ;
    Report* r = Report::load(dir);
    r->dump(); 

    return 0 ;
}
