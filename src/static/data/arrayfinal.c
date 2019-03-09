#include<stdio.h>
int arr[5], count=0;
void initialise()
{
	int i;
	for(i = 0; i<5; i++)
	{
		arr[i]= 0;
	}
}

int insert_rear(int num)
{	
	if(count<5)		
		{		
			int i=0;
			for(i =0; i<5;i++)
			{
				if(arr[i]==0)	
				{
					arr[i] = num ;
					count++;
					break;		
				}			
			}
		}
	else
		{
		printf("List is full\n");
		}
			

}
int counter()
{	
	int i =0, t=0;
	while(arr[i]!=0)
	{ 
	t++;
	i++;	
	}
	return t;
}
void insert_front(int num)
{
	int temp, temp1;
	int t=counter();
	if(t<5)
	{
		temp= num;
		int i=0;
		for (i =0; i<=t; i++)
		{
			temp1 = arr[i];
			arr[i]= temp;
			temp= temp1;

		}
		
	}
	else
	{
	printf("List is full.\n");
	}
	count++;
}


void insertpos(int num, int pos)
{	count++;
	int temp= num, temp1, i;
	int t = counter();
	pos= pos-1;
	if(pos>=0 && pos<5)
	{
		if(t<5)
		{
			for(i =0; i<=t; i++)
			{ 
				if(pos==0)
				{ 
					temp1 = arr[i];
					arr[i]= temp;
					temp= temp1;
				}
							
				else
				{	
					if(i==pos)
					{	if(arr[pos]!=0)
						{
							int j =i;
							while(j<5)
							{
								temp1 = arr[j];
								arr[j]= temp;
								temp= temp1;
								//printf(" element %d \n", arr[i]);
								j++;
							}
						}
						else if(arr[i]==0)	
						{	
							arr[i] = temp ;
							break;		
						}
										
					}
					else {continue ;}				
				}
			}
		}
	
		else
		{printf("List is full.\n");}
	}
	else
	{printf("Invalid position\n");}
}

void delete_rear()
{
	int t = counter();
	int i=0;
	for(i =0 ; i<=t; i++)
	{
		if(i==t)
		{
		arr[i]=0;
		}
	}
	count--;

}
void delete_front()
{
	int t = counter();
	int i=0;	
	for (i = 0; i<=t; i++)
		{
			if(i==t)
				{ 
				arr[i]=0;
				}
			else	
			{
				arr[i]= arr[i+1];
				
			}
		}

	count--;
}


void delete_mid(int pos)
{	

	int i=0;
	int t = counter();
	pos= pos-1;
	if(pos>=0 && pos<5)
	{
		if(t<=5)
		{
			for(i =0; i<=t; i++)
			{ 
				if(pos==0)
				{ 
					if(i==t)
					{ 
					arr[i]=0;
					}
					else	
					{
						arr[i]= arr[i+1];
				
					}
				}
							
				else
				{	
					if(i==pos)
					{	if(arr[pos]!=0)
						{
							int j =i;
							while(j<5)
							{
								if(i==t)
								{ arr[i]=0;}
								else	
								{
									arr[i]= arr[i+1];
								}
								j++;
							}
						}
						else if(arr[i]==0)	
						{	
							arr[i] = 0;
							break;		
						}
										
					}
					else {continue ;}				
				}
			}
		}
	
		else
		{printf("List is full.\n");}
	}
	else
	{printf("Invalid position\n");}
	count--;	
}
void display()
{	
	int i=0;
	int t=counter();
	if(t ==5)
	{
		for(i=0;i<5; i++)
		{printf("%d\n", arr[i]);}
	}
	else
	{
		for(i=0;i<t; i++)
		{printf("%d\n", arr[i]);}
	}
}

void insertionsort()
{
	int i,value=0,hole=0, length;
	length= counter();
	for(i=1;i<length;i++)
	{
		value=arr[i];
		hole=i;
		while(hole>0 && arr[hole-1]>value)
		{
		arr[hole]=arr[hole-1];
		hole=hole-1;	
		}
		arr[hole]=value;
	}

	/*int i, j ,tmp; 
	for (i = 1; i < length; i++) 
	{ 
    	j = i; 
		while (j > 0 && arr[j - 1] > arr[j])
   		{ 
			tmp = arr[j]; 
		   	arr[j] = arr[j - 1]; 
			arr[j - 1] = tmp; 
    		j--; 
  		}
	}*/
}

int main()
{	
	initialise();
	int choice = 20 ;
	 	
	 while (choice !=9 )
	 {	
	 	printf("1. Insert at the front.\n");
	 	printf("2. Insert at the rear.\n");
	 	printf("3. Delete at the front.\n");
	 	printf("4. Delete at the rear.\n");
	 	printf("5. Display.\n");
	 	printf("6. Insert at position.\n");
		printf("7. Delete at position.\n");
	 	printf("8. To sort by insertion sort.\n");
	 	printf("9. To exit.\n");
		int num =0,pos =0;
	 	printf("Enter choice:\n");
	 	scanf("%d", &choice);
	 	switch(choice)
	 	{
	 		case 1: 
				printf("Enter the number:\n");
				scanf("%d", &num);
				insert_front(num) ;
				break ;
	 		case 2: 
				printf("Enter the number:\n");
				scanf("%d", &num);
				insert_rear(num) ;
				break ;
	 		case 3: 
				delete_front(num) ;
				break ;
	 		case 4: 
				delete_front(num) ;
				break ;
	 		case 5 : 
				display();
				break;
	 		case 6: 
				printf("Enter the number:\n");
				scanf("%d", &num);
				printf("Enter the position:\n");
				scanf("%d", &pos);
				insertpos(num,pos);
				break;
	 		case 7: 
				printf("Enter the position:\n");
				scanf("%d", &pos);
				delete_mid(pos);
				break;
			case 8:
				insertionsort();
				break ;
			case 9:
				break;
	 		}
	 			 	
	 	
	 	}

	


}
