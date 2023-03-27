"""
test accuracy

Input: "output.txt" by my AI and "std_output.txt"
Output: word and sentence accuracy

"""

def test(output: str, std_output: str):
    with open(output, "r", encoding="gbk") as f1:
        file1 = f1.readlines()
    with open(std_output, "r", encoding="utf-8") as f2:
        file2 = f2.readlines()

    if len(file1) != len(file2):
        print(f"[wrong file length]: , file1.len = {len(file1)}, file2.len = {len(file2)}")

    word_cnt = 0 
    word_correct = 0
    line_correct = 0

    for i in range(0, min(len(file1), len(file2))):
        line1 = file1[i]
        line2 = file2[i]
        if len(line1) != len(line2):
            print(f"[wrong line[{i}] length]: line1.len = {len(line1)}, line2.len = {len(line2)}")
            continue
        word_cnt += min(len(line1), len(line2))
        flag = True
        for j in range(0, min(len(line1), len(line2))):
            if line1[j] == line2[j]:
                word_correct += 1
            else:
                flag = False
        if flag == True:
            line_correct += 1
                
    print(f"[word accuracy] = {word_correct / word_cnt}, [sentence accuracy] = {line_correct / len(file1)}")
