
Código en MAXIMA/PYTHON/Python para calcular centroide de un triángulo (Problema 3 Sección 1.1.6).
---------------------------------------------------------------------------------------------------
import numpy as np

def cal_centroide(a, b, c):
    vector_a = np.array(a)
    vector_b = np.array(b)
    vector_c = np.array(c)
    centroide = (vector_a + vector_b + vector_c) / 3.0
    return centroide

vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])
vector_c = np.array([7, 8, 9])

resul_centroide = cal_centroide(vector_a, vector_b, vector_c)

print("El vector posición del centroide G es:", resul_centroide)

-----------------------------------------------------------------------------------------
Código en MAXIMA/PYTHON que reproduzca el ejercicio 6 de la Sección 1.2.7
-----------------------------------------------------------------------------------------
from sympy import symbols, sqrt, acos, pi, Matrix

A)
---
a = Matrix([1, 2, 3])
b = Matrix([4, 5, 6])
c = Matrix([3, 2, 1])
d = Matrix([6, 5, 4])

res1 = a + b + c + d
res2 = a + b - c - d
res3 = a - b + c - d
res4 = -a + b - c + d

print("Resultado 1:", res1)
print("Resultado 2:", res2)
print("Resultado 3:", res3)
print("Resultado 4:", res4)


B)
--
# Calcular ángulos
ang_a_e1 = acos(a.dot([1, 0, 0]) / a.norm())
ang_b_e2 = acos(b.dot([0, 1, 0]) / b.norm())
ang_c_e3 = acos(c.dot([0, 0, 1]) / c.norm())
ang_d_e1 = acos(d.dot([1, 0, 0]) / d.norm())

print("Ángulo a con e1:", ang_a_e1)
print("Ángulo b con e2:", ang_b_e2)
print("Ángulo c con e3:", ang_c_e3)
print("Ángulo d con e1:", ang_d_e1)


C)
--
# Magnitudes de vectores
mag_a = a.norm()
mag_b = b.norm()
mag_c = c.norm()
mag_d = d.norm()

print("Magnitud de a:", mag_a)
print("Magnitud de b:", mag_b)
print("Magnitud de c:", mag_c)
print("Magnitud de d:", mag_d)

D)
--
ang_a_b = acos(a.dot(b) / (mag_a * mag_b))
ang_c_d = acos(c.dot(d) / (mag_c * mag_d))

print("Ángulo entre a y b:", ang_a_b)
print("Ángulo entre c y d:", ang_c_d)

E)
--
proj_a_sobre_b = a.dot(b) / mag_b * (b / mag_b)
print("Proyección de a sobre b:", proj_a_sobre_b)

F)
--
# Verificar coplanaridad
if a.cross(b).dot(c.cross(d)) == 0:
    print("Los vectores a, b, c, d son coplanares.")
else:
    print("Los vectores a, b, c, d no son coplanares.")

G)
--
    resultado_g = (a + b).dot(c + d)
print("(a+b)*(c+d):", resultado_g)
    
H)
--
# Producto punto y ángulos con d
prod_a_b = a.dot(b)
prod_b_c = b.dot(c)
prod_c_d = c.dot(d)

ang_a_b_d = acos(prod_a_b / (mag_a * mag_b))
ang_b_c_d = acos(prod_b_c / (mag_b * mag_c))
ang_c_d_d = acos(prod_c_d / (mag_c * mag_d))

print("Producto a*b:", prod_a_b)
print("Producto b*c:", prod_b_c)
print("Producto c*d:", prod_c_d)

print("Ángulo entre a*b y d:", ang_a_b_d)
print("Ángulo entre b*c y d:", ang_b_c_d)
print("Ángulo entre c*d y d:", ang_c_d_d)

I)
--
# Producto cruz y producto punto
prod_c_a_b = c.dot(a.cross(b))
print("c*(a*b):", prod_c_a_b)
