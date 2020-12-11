def printLcs(flag, a, i, j, sent, index):

    if i == 0 or j == 0:
        return
    if flag[i][j] == 'OK':
        printLcs(flag, a, i-1, j-1, sent, index)
        sent.append(a[i-1])
        index.append(i-1)
    elif flag[i][j] == 'Left':
        printLcs(flag, a, i, j-1, sent, index)
    else:
        printLcs(flag, a, i-1, j, sent, index)


def longSubSeq(str1, str2):
    sent, index = [], []
    len1 = len(str1)
    len2 = len(str2)
    longest = 0
    c = [[0 for i in range(len2+1)]for i in range(len1+1)]
    flag = [[0 for i in range(len2+1)]for i in range(len1+1)]
    for i in range(len1+1):
        for j in range(len2+1):
            if i == 0 or j == 0:
                c[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                c[i][j] = c[i-1][j-1]+1
                flag[i][j] = 'OK'
                longest = max(longest,c[i][j])
            elif c[i][j-1] > c[i-1][j]:
                c[i][j] =c[i][j-1]
                flag[i][j] = 'Left'
            else:
                c[i][j] =c[i-1][j]
                flag[i][j] = 'UP'
    printLcs(flag,str1,len1,len2, sent, index)
    return longest, sent, index


def match_1(sent1, sent2, percent=0.5):
    list1, list2 = set(sent1.split()), set(sent2.split())
    common = [i for i in list1 if i in list2]
    score = len(common) / len(list1)
    if score > percent:
        print(score)
        return True
    else:
        return False


def match_2(sent1, sent2, percent=0.5):
    if sent1 not in ' ' * 10:
        list1, list2 = sent1.split(), sent2.split()
        longest, sent, index = longSubSeq(list2, list1)
        score = longest / len(list1)
        if score > percent:
            new_list2 = ' '.join(list2[index[0]: index[-1] + 1])  # TODO narrow it
            # print(score)
            # print(new_list2)
            # print(sent2)
            return score, new_list2
        else:
            return -1, ''
    else:
        return -1, ''


if __name__ == '__main__':
    a = ['我的', 'b', '你', 'd']
    b = ['b', 'c', '我的', '你']
    longest, sent, indexprint = longSubSeq(a, b)
    print(longest, sent, indexprint)
