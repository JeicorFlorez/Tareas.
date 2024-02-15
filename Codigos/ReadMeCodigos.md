Este directorio albergará los códigos desarrollados durante el curso como parte de las tareas y asignaciones
from sympy import symbols, sqrt, acos, pi, Matrix

e1, e2, e3 = symbols('e1 e2 e3')
a = Matrix([1, 2, 3])
b = Matrix([4, 5, 6])
c = Matrix([3, 2, 1])
d = Matrix([6, 5, 4])

resultado_1 = a + b + c + d
resultado_2 = a + b - c - d
resultado_3 = a - b + c - d
resultado_4 = -a + b - c + d

print("a + b + c + d:", resultado_1)
print("a + b - c - d:", resultado_2)
print("a - b + c - d:", resultado_3)
print("-a + b - c + d:", resultado_4)

angulo_a_e1 = acos(a.dot(Matrix([1, 0, 0])) / (a.norm()))
angulo_b_e2 = acos(b.dot(Matrix([0, 1, 0])) / (b.norm()))
angulo_c_e3 = acos(c.dot(Matrix([0, 0, 1])) / (c.norm()))
angulo_d_e1 = acos(d.dot(Matrix([1, 0, 0])) / (d.norm()))

print("Ángulo entre a y ^e1:", angulo_a_e1)
print("Ángulo entre b y ^e2:", angulo_b_e2)
print("Ángulo entre c y ^e3:", angulo_c_e3)
print("Ángulo entre d y ^e1:", angulo_d_e1)

magnitud_a = a.norm()
magnitud_b = b.norm()
magnitud_c = c.norm()
magnitud_d = d.norm()

print("Magnitud de a:", magnitud_a)
print("Magnitud de b:", magnitud_b)
print("Magnitud de c:", magnitud_c)
print("Magnitud de d:", magnitud_d)

angulo_entre_a_y_b = acos(a.dot(b) / (a.norm() * b.norm()))
angulo_entre_c_y_d = acos(c.dot(d) / (c.norm() * d.norm()))

print("Ángulo entre a y b:", angulo_entre_a_y_b)
print("Ángulo entre c y d:", angulo_entre_c_y_d)

proyeccion_a_sobre_b = a.dot(b) / b.norm() * (b / b.norm())
print("Proyección de a sobre b:", proyeccion_a_sobre_b)

if a.cross(b).dot(c.cross(d)) == 0:
    print("Los vectores a, b, c, d son coplanares.")
else:
    print("Los vectores a, b, c, d no son coplanares.")

resultado_g = (a + b).dot(c + d)
print("(a+b)*(c+d):", resultado_g)

producto_a_b = a.dot(b)
producto_b_c = b.dot(c)
producto_c_d = c.dot(d)

angulo_a_b_con_d = acos(producto_a_b / (a.norm() * b.norm()))
angulo_b_c_con_d = acos(producto_b_c / (b.norm() * c.norm()))
angulo_c_d_con_d = acos(producto_c_d / (c.norm() * d.norm()))

print("Producto a*b:", producto_a_b)
print("Producto b*c:", producto_b_c)
print("Producto c*d:", producto_c_d)

print("Ángulo entre a*b y d:", angulo_a_b_con_d)
print("Ángulo entre b*c y d:", angulo_b_c_con_d)
print("Ángulo entre c*d y d:", angulo_c_d_con_d)

producto_c_a_b = c.dot(a.cross(b))
print("c*(a*b):", producto_c_a_b)
