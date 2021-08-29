#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:43:48 2021

@author: Rashmi Kethireddy
"""
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
torch.manual_seed(400)

class SpectralRule(nn.Module):
    def __init__(self, in_units, out_units, **kwargs):
        super().__init__(**kwargs)
        
        
        self.in_units, self.out_units = in_units, out_units
        self.lin1=nn.Linear(self.in_units, self.out_units)



    def forward(self,A, X):
        I = torch.eye(*A.shape[1:2])
        A_hat = A + I

        D = torch.sum(A_hat, axis=0)
        D_inv = D**-0.5
        D_inv = torch.diag(D_inv)

        A_hat = torch.tensor(D_inv * A_hat * D_inv,dtype=torch.float32)
        aggregate =torch.matmul(A_hat,X.t())
        
        propagate = F.relu(self.lin1(aggregate))
        
        return propagate
'''
def temp(A,X):
        I = torch.eye(*A.shape)
        A_hat = A.copy() + I
        A=A_hat * X
        D = np.array(np.sum(A, axis=0))[0]
        D = np.matrix(np.diag(D))
        D**-1 * A
'''
      
def main():
    #Intialize dummy A
    #Run the code for testing
    A=torch.tensor(np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1], 
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
    ))
    X=torch.tensor([
           [ 0.,  0.],
           [ 1., -1.],
           [ 2., -2.],
           [ 3., -3.]
        ])
    sr_model=SpectralRule(4,2)
    pr=sr_model(A,X)
    print(pr)
    
    
    
    
    
    

if __name__=='__main__':
    main()
