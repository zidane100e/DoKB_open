'''
file to interprete simple exepressions like calculation
It assumes that int numbers are operated with '+' & '-'
You can expand operations based on this
'''

import re

def simple_cal(exp1):
    if type(exp1) == str:
        temp = []
        for x in re.findall('(\d+|\D)', exp1):
            x2 = x.strip()
            if x2 is '':
                continue
            if not x2 in ['+', '-']:
                temp.append(int(x2))
            else:
                temp.append(x2)
        exp1 = temp
    arr1 = exp1
    if len(arr1) == 0 :
        return 0
    elif len(arr1) == 1:
        return arr1[0]
    elif len(arr1) == 2:
        if arr1[0] == '+':
            return arr1[1]
        elif arr1[0] == '-':
            return -1*arr1[1]
        else:
            raise Exception(" Should be one operand and one operator")
    
    if len(arr1) % 2 == 0: # case for - 3 + 4 --> 0 - 3 + 4
        arr1 = [0] + arr1
    for ii in range(0, len(arr1), 2):
        if type(arr1[ii]) is not int:
            raise Exception("operand error : should be operand operator operand operator ...")
        if (ii is not len(arr1)-1) and (arr1[ii+1] not in ['+', '-']):
            raise Exception("operator error : should be operand operator operand operator ...")
    
    ret = arr1[0]
    if arr1[1] == '+':
        ret += arr1[2]
    else:
        ret -= arr1[2]
    arr1 = arr1[2:]
    arr1[0] = ret
    return simple_cal(arr1)
    
def cal(exp1):
    if type(exp1) == str:
        temp = []
        for x in re.findall('(\d+|\D)', exp1):
            x2 = x.strip()
            if x2 is '':
                continue
            if not x2 in ['+', '-', '(', ')']:
                temp.append(int(x2))
            else:
                temp.append(x2)
        exp1 = temp
    arr1 = exp1
    if not '(' in arr1:
        return simple_cal(arr1)
    flag_open = False
    for ii, x in enumerate(arr1):
        if x == '(' :
            flag_open = True
            open_pos = ii
        elif x == ')' and flag_open:
            flag_open = False
            closed_pos = ii
            ret = cal(arr1[open_pos+1:closed_pos])
            return cal(arr1[:open_pos] + [ret] + arr1[closed_pos+1:])

if __name__ == '__main__':
    str1 = ' 4 - 1+4+3'
    simple_cal(str1)

    a = '(3) + ((4-2))'
    cal(a)