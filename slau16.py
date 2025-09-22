import numpy as np

f = open("pituhon/Chislaki/text.txt")
n = int(f.readline())
A = []
for _ in range(n):
    A.append([float(i) for i in f.readline().split()])
b = [float(i) for i in f.readline().split()]
f.close()

A=np.array(A)
b=np.array(b)


pred=np.array([0]*n)

for k in range(1,1000):

    ksi=np.dot(A,pred) -b

    tau=(np.dot(np.dot(A,ksi),ksi)/np.dot(np.dot(A,ksi),np.dot(A,ksi)))

    x=pred-tau*ksi
    pred=x
print(x)

'''
for k in range(1,1000):
    ksi=np.dot(A,pred) -b

    tau=(np.dot(np.dot(A,ksi),ksi)/np.dot(np.dot(A,ksi),np.dot(A,ksi)))

    E=np.eye(3)

    x=np.dot((E-tau*A),pred) + tau*b
    pred=x
print(x)
'''


'''
3
1 2 3
2 1 4
3 4 1
14 16 14
'''