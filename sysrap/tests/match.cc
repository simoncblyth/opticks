// name=match ; gcc $name.cc -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>

/**
match
-------

Based on https://www.geeksforgeeks.org/wildcard-character-matching/

The second argument string can contain wildcard tokens:

`*` 
     matches with 0 or more of any char (NB '**' not supported)
`?`   
     matches any one character.
`$`
     when appearing at end of q signifies the end of s 

**/

bool match(const char* s, const char* q) // q may contain wildcard chars ? * 
{
    if (*q == '\0' && *s == '\0') return true;

    if (*q == '*' && *(q+1) != '\0' && *s == '\0') return false;  // reached end of s but still q chars coming 

    if (*q == '$' && *(q+1) == '\0' && *s == '\0' ) return true ; 

    if (*q == '?' || *q == *s) return match(s+1, q+1);  // on to next char

    if (*q == '*') return match(s, q+1) || match(s+1, q);  // '*' can match nothing or anything in s, including literal '*'

    return false;
}

int main()
{
    assert( match("hello", "hello" )); 
    assert( match("hello", "he*llo" ));  // '*' matches nothing 
    assert( match("hello", "hello*" )); 
    assert( match("hello", "hell*" )); 
    assert( match("hello", "hell?" )); 
    assert( match("hello", "?????" )); 
    assert( match("hello", "?ell?" )); 
    assert( match("hello", "he?lo" )); 
    assert( match("hello", "he*" )); 
    assert( match("hello", "he*lo" )); 

    assert(!match("hello", "????" )); 

    assert( match("he*lo", "he*lo" )); 
    assert( match("he*lo", "he*lo" )); 

    assert( match("hello", "hello$")) ; 
    assert( !match("helloworld", "hello$")) ; 

}

