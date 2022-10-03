top , right = 10, 10
corners = ((1,1), (1,top), (right, 1), (right, top))

if corners[0] == corners[1] == corners[2] == corners[3] == True:
    print("Yes")

else:
    print("No")