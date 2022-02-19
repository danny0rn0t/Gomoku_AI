class A:
    def __init__(self, var, lst):
        self.var = var
        self.lst = lst
class B(A):
    def __init__(self, child: A):
        # super().__init__()
        self.child = child

a = A(1, [2, 3, 4])
b = B(a)
c = B(a)
print(b.child.var)
print(c.child.var)
b.child.var = 5
print(b.child.var)
print(c.child.var)
