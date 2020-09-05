
# author : bwlee@kbfg.com

import re

def filter(strs, *conds):
    """ filter a string array by conditions
    
    """
    conds = [ re.compile(x) for x in conds ]
    strs2 = []
    for x in strs:
        flag = True
        for re1 in conds:
            if re1.search(x):
                flag = None
                break
        if flag:
            strs2.append(x)
    return strs2

if __name__ == '__main__':
    filters = ['/Josa$', '/Number$', '/Determiner$', '/Suffix$', '/Punctuation$',
               '/Exclamation', '/KoreanParticle']

    str = ['asd/Josa', 'asd/Noun', 'dfg/Number', 'sdf/verb', '234/adjective']

    str2 = filter(str, *filters)

    print(str2)

