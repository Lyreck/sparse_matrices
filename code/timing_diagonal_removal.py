## testing different solutions to find the fastest way to put a csr_array's diagonal to 0.

import random as rd
from scipy.sparse import csr_array, tril, triu

import time as t

n=1000000#00

def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)


print('Creating Data...')
data1,row1,col1=[], [], []
for i in range(int(n/5)):
    #print(i)
    progress(int(i/int(n/5)*100))
    data1.append(rd.randint(0,10*n))
    row1.append(rd.randint(0,n-1))
    col1.append(rd.randint(0,n-1))

csrarray1 = csr_array((data1, (row1, col1)), shape=(n,n))
csrarray1bis = csrarray1.copy()
csrarray1bisbis = csrarray1.copy()

#méthode 1: tril et triu
t0=t.time()
csrarray1 = tril(csrarray1, -1) + triu(csrarray1, 1)
print(f'\n Method {1} took {round(t.time() - t0,2)} seconds')
csrarray1.eliminate_zeros()
print(f' removal of 0s: {round(t.time() - t0,2)} seconds')

#méthode 2: .diagonal()
t0=t.time()
n,n =csrarray1bis.shape
row=[i for i in range(n)]
diagonal = csr_array((csrarray1bis.diagonal(), (row,row)), shape=csrarray1bis.shape)
csrarray1bis -= diagonal
print(f'\n Method {2} took {round(t.time() - t0,2)} seconds')
csrarray1bis.eliminate_zeros()
print(f' removal of 0s: {round(t.time() - t0,2)} seconds')

#Méthode 3: créer une matrice identité et la multiplier par array pour avoir la diagonale.
t0=t.time()
n,n =csrarray1bisbis.shape
row=[i for i in range(n)]
data = [1 for _ in range(n)]
diagonal = csr_array((data, (row,row)), shape=csrarray1bisbis.shape)
csrarray1bisbis -= csrarray1bisbis * diagonal
print(f'\n Method {3} took {round(t.time() - t0,2)} seconds')

#Méthode 4: setdiag()
t0=t.time() ### BCP PLUS LONG
csrarray1.setdiag(0)
print(f'\n Method {4} took {round(t.time() - t0,2)} seconds')
csrarray1.eliminate_zeros()
print(f' removal of 0s: {round(t.time() - t0,2)} seconds')