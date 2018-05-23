#include "Report.hpp"

void test_save()
{
    std::vector<std::string> lines ; 
    lines.push_back("hello");
    lines.push_back("world");

    Report r ; 
    r.add(lines);
    r.save("$TMP");
}


void test_load()
{
    Report* r = Report::load("$TMP");
    r->dump(); 
}



int main()
{
    test_save();
    test_load();

    return 0 ;
}
