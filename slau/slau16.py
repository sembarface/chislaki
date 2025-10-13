import numpy as np

f = open("text.txt")
n = int(f.readline())
A = []
for _ in range(n):
    A.append([float(i) for i in f.readline().split()])
b = [float(i) for i in f.readline().split()]
f.close()

A=np.array(A)
b=np.array(b)
eps=1e-10
trans=np.transpose(A)
sim=np.dot(trans,A)
bsim=np.dot(trans,b)

if np.array_equal(A,trans):
    sim=A
    bsim=b


pred=np.array([0]*n)
ksi=np.array([1]+[0]*(n-1))
while(np.linalg.norm(ksi)> eps):

    ksi=np.dot(sim,pred) -bsim

    tau=(np.dot(np.dot(sim,ksi),ksi)/np.dot(np.dot(sim,ksi),np.dot(sim,ksi)))

    x=pred-tau*ksi
    pred=x

print("Ответ: ")
print(x)

'''
for k in range(1,1000):
    ksi=np.dot(sim,pred) -bsim
    tau=(np.dot(np.dot(sim,ksi),ksi)/np.dot(np.dot(sim,ksi),np.dot(sim,ksi)))
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

otvet 1 2 3
'''


'''
3
1 2 3
0 3 5
3 5 7
-1 -3 -2

otevet 1 -1 0
'''
