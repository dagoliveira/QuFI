import os

# removing the new line characters
f = open('requirements.txt')
lines = f.readlines()

for li in lines:
    l = li.strip()
    name = l.split("=")[0]
    version = l.split("=")[1]
    p = "pip install " + name + "==" + version
    print(p)

    os.system(p)