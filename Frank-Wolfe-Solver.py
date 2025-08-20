from sympy import symbols, diff, lambdify, sin, cos, pi, exp, sqrt, Abs, log
import math
import inspect
z1, z2 = symbols('x1 x2')

#Functions 1: Optimization Benchmark Functions
f1 = (z1 + 2*z2 - 7)**2 + (2*z1 + z2 - 5)**2 # Booth Function
f2 = 0.26 * (z1**2 + z2**2) - 0.48 * z1 * z2 # Matyas Function
f3 = sin(z1 + z2) + (z1 - z2)**2 - 1.5*z1 + 2.5*z2 + 1 # McCormick Function
f4 = 2 * z1**2 - 1.05 * z1**4 + (z1**6) / 6 + z1 * z2 + z2**2 # Three-Hump Camel Function
f5 = (4 - 2.1 * z1**2 + (z1**4) / 3) * z1**2 + z1 * z2 + (-4 + 4 * z2**2) * z2**2 # Six-Hump Camel Function
f6 = z1**2 + 2 * z2**2 - 0.3 * cos(3 * pi * z1) - 0.4 * cos(4 * pi * z2) + 0.7 # Bohachevsky Function
f7 = -cos(z1) * cos(z2) * exp(-((z1 - pi)**2) - ((z2 - pi)**2)) # Easom Function
f8 = (1.5 - z1 + z1 * z2)**2 + (2.25 - z1 + z1 * z2**2)**2 + (2.625 - z1 + z1 * z2**3)**2 # Beale Function
f9 = 0.5 * ((z1**4 - 16*z1**2 + 5*z1) + (z2**4 - 16*z2**2 + 5*z2)) # Styblinski-Tang Function
f10 = sin(3 * pi * z1)**2 + (z1 - 1)**2 * (1 + sin(3 * pi * z2)**2) + (z2 - 1)**2 * (1 + sin(2 * pi * z2)**2) # Levy Function
f11 = 0.5 + (sin(z1**2 - z2**2)**2 - 0.5) / (1 + 0.001 * (z1**2 + z2**2))**2 # Schaffer Function No: 2

#Functions 2: Error Functions
x1, x2=[], []
for i in range(11):
    x1.append(i)
    x2.append(i)
def e1(x1,x2, i):
    return ((x1[i]-x1[i-10])**2+(x2[i]-x2[i-10])**2)/2 # Adapted from MSE Loss
def e2(x1, x2, i):
    return (Abs(x1[i]-x1[i-10])+Abs(x2[i]-x2[i-10]))/2 # Adapted from MBE Loss

#Functions 3: Step Size Functions
def y1(k):
    return (2/(2+k))
def y2(k):
    return (1/(1+k))
def y3(k):
    return (0.001)

#Declarations 1
f=f1
e=e1
y=y1
maxe=0.0000001
itr=5001
x1=[0]
x2=[0]
x1min, x1max = -5, 5
x2min, x2max = -5, 5
dfdz1 = diff(f, z1)
dfdz2 = diff(f, z2)
print("Function:", f)
print("Gradient1:", dfdz1)
print("Gradient2:", dfdz2)
fnum=lambdify((z1, z2), f)
grad1=lambdify((z1, z2), dfdz1)
grad2=lambdify((z1, z2), dfdz2)

#Algorithm 1: Frank-Wolfe Algorithm
for i in range(itr):
    gval1=grad1(x1[i], x2[i])
    gval2=grad2(x1[i], x2[i])
    fslin=gval1*z1+gval2*z2
    fflin=lambdify((z1, z2), fslin)
    candidates=[
        (x1min,x2min),
        (x1min, x2max),
        (x1max, x2min),
        (x1max,x2max)]
    flinvec=[fflin(z1,z2) for z1, z2 in candidates]
    s1, s2=candidates[flinvec.index(min(flinvec))]
    d1, d2=s1-x1[i], s2-x2[i]
    x1new, x2new=x1[i]+y(i)*d1, x2[i]+y(i)*d2
    x1.append(x1new), x2.append(x2new)
    if i>10:
        loss=e(x1, x2, i)
        if loss<maxe:
            break

#Outputs 1: Last n Step as Output
for n in range(10,-1,-1):
    print("x1:", f'{x1[i-n]:.3f}', "x2:", f'{x2[i-n]:.3f}', "f(x):", f'{fnum(x1[i-n], x2[i-n]):.3f}', "Iteration:", i-n)
print ("\nOptimisation done. Check results. Especially for the non-convex functions.\n")