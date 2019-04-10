index=0
for line in open("neglist.txt","r"):
    index=index+1
    path="out/dataneg"+str(index)+".txt"
    with open(path, "w") as f:
        f.writelines(line)
