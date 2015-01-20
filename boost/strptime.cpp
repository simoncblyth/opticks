
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(void)
{
    const char* tst = "Mar 9, 2010 12:55:10" ;
    const char* ifmt = "%b %d, %Y %H:%M:%S" ;
    const char* ofmt = "%c" ;

    struct tm tm;
    memset(&tm, 0, sizeof(struct tm));
    strptime( tst , ifmt , &tm );

    char buf[255];
    strftime(buf, sizeof(buf), ofmt , &tm);
    puts(buf);

    exit(EXIT_SUCCESS);
}

