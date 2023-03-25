with open("output.txt", "r", encoding="gbk") as f1:
    file1 = f1.readlines()
with open("std_output.txt", "r", encoding="utf-8") as f2:
    file2 = f2.readlines()

if len(file1) != len(file2):
    print(f"[wrong file length]: , file1.len = {len(file1)}, file2.len = {len(file2)}")
cnt = 0 
correct = 0
for i in range(0, min(len(file1), len(file2))):
    line1 = file1[i]
    line2 = file2[i]
    if len(line1) != len(line2):
        print(f"[wrong]: {i}, line1.len = {len(line1)}, line2.len = {len(line2)}")
    cnt += min(len(line1), len(line2))
    for j in range(0, min(len(line1), len(line2))):
        if line1[j] == line2[j]:
            correct += 1
print(correct / cnt)
