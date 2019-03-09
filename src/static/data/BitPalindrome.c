//LET US CHECK FOR BIT PALINDROME
#include<stdio.h>
int main()
{
    int no,noc,rev=0;
    printf("Enter The Number\n");
    scanf("%d",&no);
    noc=no;
    while(no>0)
    {

        rev=(rev<<1)+(no&1);
        no=no>>1;
    }
    printf("%d %d",rev,noc);
    if (rev==noc)
        printf("Bit Palindrome");
    else
        printf("Not Bit Palindrome");
}
