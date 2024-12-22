#comparing @ and dot for matrix product.

import time as t
from scipy.sparse import csr_array
import random as rd
import numpy as np

#  `@` took 7.66 seconds
# `dot` took 6.53 seconds
## dot SEMBLE PLUS RAPIDE.
n=1000000

def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)

print('Creating Data1...')
data1,row1,col1=[], [], []
for i in range(int(n/5)):
    data1.append(rd.randint(0,10*n))
    row1.append(rd.randint(0,n-1))
    col1.append(rd.randint(0,n-1))
    progress(int((i+1)/int(n/5)*100))


print('\n Creating Data2...')
data2,row2,col2=[], [], []
for i in range(int(n/4)):
    data2.append(rd.randint(0,10*n))
    row2.append(rd.randint(0,n-1))
    col2.append(rd.randint(0,n-1))
    progress(int((i+1)/int(n/4)*100))


csrarray1 = csr_array((data1, (row1, col1)), shape=(n,n))
csrarray2 = csr_array((data2, (row2, col2)), shape=(n,n))


t0=t.time()
csrarray3 = csrarray1 @ csrarray2
print(f'\n `@` took {round(t.time() - t0,2)} seconds')

t0=t.time()
csrarray4 = csrarray1.dot(csrarray2)
print(f'`dot` took {round(t.time() - t0,2)} seconds')


assert (
    np.all(csrarray3.indices == csrarray4.indices)
    and np.all(csrarray3.indptr == csrarray4.indptr)
    and np.allclose(csrarray3.data, csrarray4.data)
)

