#include<stdio.h>
#include<stdlib.h>
int track=0;
struct node
{
    int data;
    struct node *next;
};
struct node *first;
struct node *last;


int createarray(int len)
{
    struct node *p;
    first=malloc(sizeof(struct node *));
    first->data=89098; //assigning random junk to array
    last=first;
    for(int i=1;i<len;i++)
    {
        p=malloc(sizeof(struct node *));
        p->data=89098;
        last->next=p;
        last=p;
    }
}

int frontins(int n)
{
    struct node *temp;
    int curr, nex;
    if(first->data==89098)
    {
        first->data=n;
        printf("\nArray Element Inserted\n");
    }
    else if(first->data!=89098 && last->data==89098)
    {
        temp=first;
        curr=first->data;
        while(temp->data!=89098)
        {
          nex=(temp->next)->data;
          (temp->next)->data=curr;
          curr=nex;
          temp=temp->next;
        }
        first->data=n;
        printf("\nArray Element Inserted\n");
    }
    else
    {
        printf("\nArray Index Out Of Bounds\n");
    }
    track++;
}

int endins(int n)
{
    struct node *temp;
    int curr, nex;
    if(first->data==89098)
    {
        first->data=n;
        printf("\nArray Element Inserted\n");
    }
    else if(first->data!=89098 && last->data==89098)
    {
        temp=first;
        while(temp->data!=89098)
        {
          temp=temp->next;
        }
        temp->data=n;
        printf("\nArray Element Inserted\n");
    }
    else
    {
        printf("\nArray Index Out Of Bounds\n");
    }
    track++;
}

int frontdel()
{
    struct node *temp,*buff;
    //int curr, nex;
    if(first->data==89098)
    {
        printf("\nEmpty List\n");
    }
    else
    {
        /*temp=first;
        curr=(temp->next)->data;
        while(temp->data!=89098 || temp->next!=NULL)
        {
          nex=(temp->next->next)->data;
          temp->data=curr;
          if(temp->next->next==NULL)

          temp->next->data=nex;
          temp=temp->next;
          cur=nex;
        }*/
        temp=first;
        first=temp->next;
        free(temp);
        struct node *p;
        p=malloc(sizeof(struct node *));
        p->data=89098;
        last->next=p;
        last=p;
        printf("\nArray Element Deleted\n");
    }
    track--;
}
int enddel()
{
    struct node *temp;
    if(first->data==89098)
    {
        printf("\nEmpty List\n");
    }
    else
    {
        temp=first;
        while(temp->next->data!=89098 || temp->next!=NULL)
        {
        temp=temp->next;
        }
        temp->data=89098;
        printf("\nArray Element Deleted\n");
    }
    track--;
}

int display()
{
   struct node *temp;
    if(first->data==89098)
    {
        printf("\nEmpty List\n");
    }
    else
    {
        temp=first;
        while(temp->next->data!=89098 || temp->next!=NULL)
        {
        temp=temp->next;
        printf("%d ", temp->data);
        }
    }
}

int posins(int pos, int n)
{
    struct node *temp, *buffer;
    int i=0,curr, nex;
    if(first->data==89098)
    {
        first->data=n;
        printf("\nArray Element Inserted\n");
    }
    else if(first->data!=89098 && last->data==89098)
    {
        temp=first;
        while(temp->next->data!=89098 || temp->next!=NULL )
        {
          temp=temp->next;
          if (i==pos)
            break;
          i++;
        }
        buffer=temp;
        curr=buffer->data;
        while(buffer->data!=89098)
        {
          nex=(buffer->next)->data;
          (buffer->next)->data=curr;
          curr=nex;
          buffer=buffer->next;
        }
        temp->data=n;
        printf("\nArray Element Inserted\n");
    }
    else
    {
        printf("\nArray Index Out Of Bounds\n");
    }
    track++;
}

int posdel(int pos)
{
    struct node *temp,*buffer;
    int curr, nex,i=0;
    if(first->data==89098)
    {
        printf("\nEmpty List\n");
    }
    else
    {
        temp=first;
        while(temp->next->data!=89098 || temp->next!=NULL )
        {
          temp=temp->next;
          if (i==pos)
            break;
          i++;
        }
        buffer=temp;
        }

}

void main()
{
    int opt=15,elem=0, len=0, pos=0;
while(opt!=9)
{
printf("Enter \n1. To create an array of n elements \n2. To front insert \n3. To end insert \n4. To delete at front \n5. To delete at end \n6. To display \n7. To position insert \n8. To position delete\n9. To exit \n");
scanf("%d", &opt);
switch(opt)
{
case 1:
printf("Enter the length of array\n");
scanf("%d", &len);
createarray(len);
break;
case 2:
printf("\nEnter element:\n");
scanf("%d", &elem);
frontins(elem);
break;
case 3:
printf("\nEnter element:\n");
scanf("%d", &elem);
endins(elem);
break;
case 4:
frontdel();
break;
case 5:
enddel();
break;
case 6:
display();
break;
case 7:
printf("\nEnter element:\n");
scanf("%d", &elem);
printf("\nEnter position:\n");
scanf("%d", &pos);
posins(pos, elem);
break;
case 8:
printf("\nEnter position:\n");
scanf("%d", &pos);
posdel(pos);
break;
}
}
}
