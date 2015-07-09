#include "Report.hpp"

int main()
{
    std::vector<std::string> lines ; 
    lines.push_back("hello");
    lines.push_back("world");

    Report r ; 
    r.add(lines);
    r.save("/tmp", "TestReport.txt");
    r.save("$IDPATH/tmp", Report::name("typ","tag").c_str() );

    return 0 ;
}
