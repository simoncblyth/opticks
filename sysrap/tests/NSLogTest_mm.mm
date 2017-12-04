
#import <Foundation/Foundation.h>



void c()
{
    NSLog(@"c");
    NSLog(@"Stack trace : %@",[NSThread callStackSymbols]);
}


void b()
{
    NSLog(@"b");
    c(); 
}

void a()
{
    NSLog(@"a");
    b();
}



int main(int argc, char** argv)
{
    NSLog(@"Testing");
    a(); 
}

/*

*/


