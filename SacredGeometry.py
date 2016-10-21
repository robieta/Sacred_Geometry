# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:11:02 2016

@author: RDX
"""
import numpy as np
from math import factorial as fact
from copy import copy

class GeometricTables():
    def __init__(self):
        self.nRolls = 6
        self.MaxRollVal = 6
        self.MaxNumTriesPerRoll = int(1e3)
        
        self.nRollFact = fact(self.nRolls)
        
        self.PrimeList = [i for i in range(3,108) if self.CheckPrime(i)]
        nPrimes = len(self.PrimeList)
        self.PrimeGroups = (
                [self.PrimeList[3*i:3*(i+1)] for i in range(int(nPrimes/3))])
        
        self.OperatorList = ['+','-','*','/']
        self.nOp = len(self.OperatorList)
        self.OperatorStrength = [1,1.1,2,2.1]
    
    def GenerateTable(self):
        np.random.seed(0)
        
        def PadStr(StrIn,target):
            for k in range(target - len(StrIn)):
                StrIn += ' '
            return StrIn
        
        #Table of possible roll combinations
        RollTable = []

        CurrentRoll = np.array([1 for m in range(self.nRolls)])

        RollTable.append(np.copy(CurrentRoll).tolist())
        while True:
#        for TempVal in range(6):
            for i in range(self.nRolls-1,-1,-1):
                if CurrentRoll[i] < self.MaxRollVal:
                    IncIndex = i
                    break
            else:
                break
            CurrentRoll[IncIndex] += 1
            CurrentRoll[IncIndex+1:] = CurrentRoll[IncIndex]
            RollTable.append(np.copy(CurrentRoll).tolist())

        # Determine the degeneracy of each roll
        omegaList = []
        for i in RollTable:
            CountList = [0 for m in range(self.MaxRollVal)]
            for j in i:
                CountList[j-1] += 1
            omegaList.append(
                         self.nRollFact/np.prod([fact(m) for m in CountList]))
        
        # Ensure degeneracy has been calculated correctly
        assert int(np.sum(omegaList)) == self.MaxRollVal ** self.nRolls

        
        SolnFound = [[False for m in range(len(self.PrimeGroups))] 
                     for mm in range(len(RollTable))]
        Soln = [['' for m in range(len(self.PrimeGroups))] 
                for mm in range(len(RollTable))]
        
        for ide,i in enumerate(RollTable):
            for TryIndex in range(self.MaxNumTriesPerRoll):
                Expression, Value = self.RevPol(RollTable[ide])
                if not np.isnan(Value):
                    if Value >= self.PrimeList[0]:
                        if Value % 1 == 0:
                            if Value in self.PrimeList:
                                InfixStr = self.RevPolToInfixStr(Expression)
                                if np.round(eval(InfixStr),10) != Value:
                                    print(Value)
                                    print(Expression)
                                    print(InfixStr)
                                    print(eval(InfixStr))
                                assert np.round(eval(InfixStr),10) == Value
                                
                                for jde,j in enumerate(self.PrimeGroups):
                                    if Value in j:
                                        LevelIndex = jde
                                CandidateSolnStr = InfixStr + '=' + str(int(Value))
                                if not SolnFound[ide][LevelIndex]:
                                    SolnFound[ide][LevelIndex] = True
                                    Soln[ide][LevelIndex] = CandidateSolnStr
                                elif len(CandidateSolnStr) < len(Soln[ide][LevelIndex]):
                                    Soln[ide][LevelIndex] = CandidateSolnStr
                if np.all(SolnFound[ide]):
                    print(PadStr(str(ide+1) + '/' + str(len(RollTable)),            
                             2+2*(len(str(len(RollTable))))) + 'Roll: ' 
                             + ' ' * 5 + str(i) + ' ' * 5
                             + 'Number of Successes: '
                             + str(np.sum(SolnFound[ide])) + '/' 
                             + str(len(self.PrimeGroups)))
                    break
            else:
                print(PadStr(str(ide+1) + '/' + str(len(RollTable)),            
                             2+2*(len(str(len(RollTable))))) + 'Roll: ' 
                             + ' ' * 5 + str(i) + ' ' * 5
                             + 'Number of Successes: '
                             + str(np.sum(SolnFound[ide])) + '/' 
                             + str(len(self.PrimeGroups)))
        
        MaxSolnLen = 0
        for i in Soln:
            for j in i:
                if len(j) > MaxSolnLen:
                    MaxSolnLen = len(j)
                
        SuccessCount = [0 for i in range(len(self.PrimeGroups))]
        for ide,i in enumerate(RollTable):
            for jde in range(len(self.PrimeGroups)):
                if SolnFound[ide][jde]:
                    SuccessCount[jde] += omegaList[ide]
        pSuccess = [i/self.MaxRollVal ** self.nRolls for i in SuccessCount]
        
        OutFile = 'SacredGeoOutput.txt'
        with open(OutFile,'w') as txt:
            txt.write(PadStr('Level:',len(str(RollTable[0]))+2))
            for i in range(1,1+len(self.PrimeGroups)):
                if i==1:
                    suffix='st'
                elif i==2:
                    suffix='nd'
                elif i==3:
                    suffix=='rd'
                else:
                    suffix='th'
                txt.write(PadStr(str(i)+suffix,MaxSolnLen+1)+'|')
            txt.write('\n')
            
            
            txt.write(PadStr('Success Rate:',len(str(RollTable[0]))+2))
            for i in pSuccess:
                txt.write(PadStr('{:.1f}'.format(i*100) + '%',MaxSolnLen+1)+'|')
            txt.write('\n')
            
            txt.write(PadStr('Targets:',len(str(RollTable[0]))+2))
            for i in self.PrimeGroups:
                txt.write(PadStr(str(i),MaxSolnLen+1)+'|')
            txt.write('\n')
            
            txt.write(PadStr('',len(str(RollTable[0]))+2))
            txt.write('-'*(MaxSolnLen+2)*len(self.PrimeGroups))
            txt.write('\n')
            
            for ide,i in enumerate(RollTable):
                txt.write(str(i) + '  ')
                for j in Soln[ide]:
                    txt.write(PadStr(j,MaxSolnLen+1)+'|')
                txt.write('\n')
                    
        
    def RevPol(self,Roll):
        OperatorChoice = [self.OperatorList[i] for i in 
                          np.random.randint(0,self.nOp,self.nRolls-1)]
        PermutedRoll = np.random.permutation(Roll).tolist()
        Expression = [PermutedRoll.pop(),PermutedRoll.pop()]
        for i in OperatorChoice:
            PermutedRoll.append(i)
        np.random.shuffle(PermutedRoll)
        for i in PermutedRoll:
            Expression.append(i)
        Value = self.RevPolInterp(Expression)
        return Expression, Value

        
    def RevPolInterp(self,Expression):
        Stack = []
        Output = 0.
        for i in Expression:
            if isinstance(i,str):
                if len(Stack) < 2:
                    Output = np.NaN
                    break
                else:
                    if i == '+':
                        Stack.append(Stack.pop() + Stack.pop())
                    elif i == '-':
                        Stack.append(-Stack.pop() + Stack.pop())
                    elif i == '*':
                        Stack.append(Stack.pop() * Stack.pop())
                    elif i == '/':
                        B = Stack.pop()
                        if B == 0:
                            Output = np.NaN
                            break
                        else:
                            Stack.append(Stack.pop() / B)
            else:
                Stack.append(float(i))
        
        if len(Stack) != 1:
            Output = np.NaN
        
        if not np.isnan(Output):
            Output = Stack.pop()
        
        return Output
        
    def RevPolToInfixStr(self,Expression):
        Stack = []
        for i in Expression:
            if isinstance(i,str):
                if len(Stack) < 2:
                    Output = 'Invalid RPN'
                    break
                else:
                    B=Stack.pop()
                    A=Stack.pop()
                    OpIndex = [m for m in range(self.nOp) if i==self.OperatorList[m]][0]
                    OpStrength = self.OperatorStrength[OpIndex]

                    if len(A) == 1:
                        LeftSide = A[0]
                    else:
                        OpIndexA = [m for m in range(self.nOp) if A[1]==self.OperatorList[m]][0]
                        OpStrengthA = self.OperatorStrength[OpIndexA]

                        LeftSide = A[0] + A[1] + A[2]
                        if (OpStrength > OpStrengthA or 
                                        (OpStrength == OpStrengthA 
                                         and (i=='/' or i=='-'))):
                            LeftSide = '(' + LeftSide + ')'
                    if len(B) == 1:
                        RightSide = B[0]
                    else:
                        OpIndexB = [m for m in range(self.nOp) if B[1]==self.OperatorList[m]][0]
                        OpStrengthB = self.OperatorStrength[OpIndexB]

                        RightSide = B[0] + B[1] + B[2]
                        if (OpStrength > OpStrengthB or 
                                        (OpStrength == OpStrengthB 
                                         and (i=='/' or i=='-'))):
                            RightSide = '(' + RightSide + ')'
                    Stack.append([LeftSide,i,RightSide])
                    
            else:
                Stack.append([str(i)])
        Output = Stack[0][0] + Stack[0][1] + Stack[0][2]
        return Output
    
    def CheckPrime(self,num):
        MaxCheck = int(np.floor(np.sqrt(num)))
        PrimeBool = True
        if num > 2:
            for i in range(2,MaxCheck + 1):
                if num%i==0:
                    PrimeBool = False
                    break
        return PrimeBool
    
GeometricTables().GenerateTable()