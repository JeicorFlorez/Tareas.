
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

ang_a_x = acos(a.dot(Matrix([1, 0, 0])) / (a.norm()))
ang_b_y = acos(b.dot(Matrix([0, 1, 0])) / (b.norm()))
ang_c_z = acos(c.dot(Matrix([0, 0, 1])) / (c.norm()))
ang_d_x = acos(d.dot(Matrix([1, 0, 0])) / (d.norm()))

print("Ángulo a con x:", ang_a_x)
print("Ángulo b con y:", ang_b_y)
print("Ángulo c con z:", ang_c_z)
print("Ángulo d con x:", ang_d_x)

mag_a = a.norm()
mag_b = b.norm()
mag_c = c.norm()
mag_d = d.norm()

print("Magnitud a:", mag_a)
print("Magnitud b:", mag_b)
print("Magnitud c:", mag_c)
print("Magnitud d:", mag_d)

ang_a_b = acos(a.dot(b) / (a.norm() * b.norm()))
ang_c_d = acos(c.dot(d) / (c.norm() * d.norm()))

print("Ángulo a y b:", ang_a_b)
print("Ángulo c y d:", ang_c_d)

proj_a_sobre_b = a.dot(b) / b.norm() * (b / b.norm())
print("Proyección de a sobre b:", proj_a_sobre_b)

if a.cross(b).dot(c.cross(d)) == 0:
    print("Vectores a, b, c, d son coplanares.")
else:
    print("Vectores a, b, c, d no son coplanares.")

res_g = (a + b).dot(c + d)
print("(a+b)*(c+d):", res_g)

prod_a_b = a.dot(b)
prod_b_c = b.dot(c)
prod_c_d = c.dot(d)

ang_a_b_con_d = acos(prod_a_b / (a.norm() * b.norm()))
ang_b_c_con_d = acos(prod_b_c / (b.norm() * c.norm()))
ang_c_d_con_d = acos(prod_c_d / (c.norm() * d.norm()))

print("Producto a*b:", prod_a_b)
print("Producto b*c:", prod_b_c)
print("Producto c*d:", prod_c_d)
print("Ángulo entre a*b y d:", ang_a_b_con_d)
print("Ángulo entre b*c y d:", ang_b_c_con_d)
print("Ángulo entre c*d y d:", ang_c_d_con_d)

prod_c_a_cruz_b = c.dot(a.cross(b))
print("c*(a*b):", prod_c_a_cruz_b)

