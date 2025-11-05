from math import *
eps=1e-10

def f(x):
    '''return (sin(x) - (x-2)**3)'''
    return (5-(x-2)**2)**2

def diff(x):
    return (f(x+eps)-f(x))/eps

def fi(x):
    return (x-f(x)/diff(x))

#Метод Ньютона
x0=0
x=fi(x0)
c=0
while(abs(x-x0)>eps):
    x0=x
    x=fi(x0)
    c+=1
print('Обычный метод Ньютона')
print(f'Ответ {x}')
print(f'Итераций {c}')
print()

#Метод Вегстейна
c=0
x0=0
x1=fi(x0)
x0_=x0
x1_=x1
x2=fi(x1_)
while(abs(x2-x1_)>eps):
    x2_=(x2*x0_-x1*x1_)/(x2+x0_-x1-x1_)
    x0=x1
    x0_=x1_
    x1=x2
    x1_=x2_
    x2=fi(x1_)
    c+=1
x=x2
print('Метод Вегстейна')
print(f'Ответ {x}')
print(f'Итераций {c}')
