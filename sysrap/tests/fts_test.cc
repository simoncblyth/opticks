// name=fts_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name /tmp/NPFold_test
// https://stackoverflow.com/questions/12609747/traversing-a-filesystem-with-fts3

#include <cstdlib>
#include <cstdio>
#include <sys/types.h>
#include <fts.h>
#include <cstring>
#include <errno.h>

int compare(const FTSENT** one, const FTSENT** two)
{
    return (strcmp((*one)->fts_name, (*two)->fts_name));
}
void indent(int i)
{ 
    for(; i > 0; i--) printf("   ");
}

void dump(char* base)
{
    char* path[2] {base, nullptr};

    FTS* fs = fts_open(path,FTS_COMFOLLOW|FTS_NOCHDIR,&compare);
    if(fs == nullptr) return ; 

    FTSENT* node = nullptr ;
    while((node = fts_read(fs)) != nullptr)
    {
        switch (node->fts_info) 
        {
            case FTS_D :
                break;
            case FTS_F :
            case FTS_SL:
                indent(node->fts_level);
                printf("%20s %s \n", node->fts_name, node->fts_path+strlen(base)+1 );
                break;
            default:
                break;
        }
    }
    fts_close(fs);
}

int main(int argc, char** argv)
{
    if (argc<2)
    {
        printf("Usage: %s <path-spec>\n", argv[0]);
        exit(255);
    }

    dump(argv[1]); 

    return 0;
}


