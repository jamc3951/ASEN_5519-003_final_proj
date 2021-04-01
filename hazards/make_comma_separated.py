
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]
for f in onlyfiles:
    if "new" not in f and ".py" not in f:
        file1 = open(f, 'r')
        new_file_name = f[:-4] + "_new.txt"
        print(new_file_name)

        new_file = open(new_file_name, 'w')
        Lines = file1.readlines()

        new_file_lines = []
        for line in Lines:
            line = line.strip()
            new_line = ""
            for c in line:
                new_line += c + ","
            new_line = new_line[:-1]
            new_line += "\n"
            new_file_lines.append(new_line)
            print(new_line)

        new_file.writelines(new_file_lines)
        new_file.close()
        file1.close()