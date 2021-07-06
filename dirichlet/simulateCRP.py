#!/usr/bin/env python3

from numpy.core.fromnumeric import cumsum
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt

### Change the value below to simulate larger permutations ###
SizeOfTheRestaurant=3500

# Dirichlet parameter
alpha = 200

def InsertAbove(List,x,i):
    WhereIs_i= List.index(i)
    return List[:WhereIs_i]+[x]+List[WhereIs_i:]

def Restaurant(n):

    rng = default_rng()

    mTableCounts = np.zeros(shape=SizeOfTheRestaurant, dtype=int)
    mTableProbability = np.zeros(shape=SizeOfTheRestaurant, dtype=float)
    
    SmallRestaurant=[[0]]
    mTableCounts[0] = 1
    
    for nCustomer in range(2,n+1):

        totalSum = 0.0
        for i in range(len(SmallRestaurant)):
            mTableProbability[i] = mTableCounts[i]/(alpha + nCustomer + 1)
            totalSum += mTableProbability[i]

        mTableProbability /= totalSum

        mNewTableProbability = alpha/(alpha + nCustomer + 1)

        NumberOfTables=len(SmallRestaurant)

        sortedTable = np.argsort(mTableProbability)

        t = rng.random()
        if t < mNewTableProbability:
            SmallRestaurant.append([nCustomer])
            mTableCounts[len(SmallRestaurant)] = 1
        else:
            t = rng.random()
            NewPlace = -1
            cumulativeSum = 0.0
            for i in range(SizeOfTheRestaurant):
                if t >= cumulativeSum and t < (cumulativeSum + mTableProbability[sortedTable[i]]):
                    NewPlace = sortedTable[i]
                    mTableCounts[NewPlace] += 1
                cumulativeSum += mTableProbability[sortedTable[i]]

            assert(NewPlace >= 0)
            assert(NewPlace < len(SmallRestaurant))
            SmallRestaurant[NewPlace].append(nCustomer)
        
    return SmallRestaurant


def Center(List,x,A,B):
    if len(List)==1:
        return List+[x]
    else:
        FirstList=List[0]
        SecondList=List[1]
        n=len(FirstList)
        m=len(SecondList)
        return [FirstList,x]+Center(List[1:],x+A+B*n+B*m,A,B)
    
theta = np.linspace(0, 2*np.pi, 40)
x = np.cos(theta)
y = np.sin(theta)

def DrawTables(Center,A,B):
    if len(Center)==2:
        x=Center[1]+B*np.cos(theta)*len(Center[0])
        y=B*np.sin(theta)*len(Center[0])
        plt.plot(x,y)
    else:
        x=Center[1]+B*np.cos(theta)*len(Center[0])
        y=B*np.sin(theta)*len(Center[0])
        plt.plot(x,y)
        DrawTables(Center[2:],A,B)
        
plt.axis("equal")
plt.axis("off")

R=Restaurant(SizeOfTheRestaurant)
C=Center(R,0,10,0.5)
DrawTables(C,4,0.5)
plt.show()
print('Size of the restaurant =' +str(SizeOfTheRestaurant))
print('Number of Occupied Tables = ' + str(len(R)))
print('Sizes of tables =' +str([len(Table) for Table in R]))