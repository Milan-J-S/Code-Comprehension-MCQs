//LINKED LIST
#include<stdio.h>
struct list
{ int data;
struct list *link;
}*head;
void insert(int data)
{
    struct list *node;
    node=(struct list*)malloc(sizeof(struct list));
    node->data=data;
    node->link=head;
    head=node;
}
void count(int ele)
{
    int elcount=0;
    struct list*temp;
    if (head==NULL)
        {
            printf("\n EMPTY LIST\n");
            return;
        }
        else
        {
            temp=head;
            while(temp!=NULL)
            {
                if (temp->data==ele)
                    elcount++;
                temp=temp->link;
            }
        }
        printf("The element %d is occurring %d times\n", ele,elcount);
}
void display()
{
    struct list*temp;
    if(head==NULL)
    {
        printf("Not Possible");
        return;
    }
    else
    {
        temp=head;
        while(temp!=NULL)
        {
            printf("%d\n",temp->data);
            temp=temp->link;
        }
    }
}
int main()
{
    int data,ele,ch;
    while(1)
    {

        printf("\n ENTER \n 1-INSERT \n 2-COUNT \n 3-DISPLAY \n");
        scanf("%d",&ch);
        switch(ch)
        {
            case 1:
                printf("\nEnter Data\n");
                scanf("%d",&data);
                insert(data);
                break;
            case 2:
                printf("\n Enter Element\n");
                scanf("%d",&ele);
                count(ele);
                break;
            case 3:
                display();
                break;
            default:exit(0);
        }
    } return(0);
}
