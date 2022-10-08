
a = [(1,1),(1,2),True,(1,3),True]
b = []
for i in range(len(a)):
    if type(a[i]) == tuple:
        b.append(a[i])

print(b)