// boost-;gcc tok.cc -std=c++11  -I$(boost-prefix)/include -o /tmp/tok && /tmp/tok

#include <cstdlib>
#include <string>
#include <iostream>
#include <iterator>
#include <boost/tokenizer.hpp>

int main()
{
    const char* line = "red green blue" ; 
    boost::char_separator<char> sep = {" "} ; 
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    tokenizer tok(line, sep);
    return 0 ; 
/*



  for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it)
    std::cout << *it << '\n';
*/

}
