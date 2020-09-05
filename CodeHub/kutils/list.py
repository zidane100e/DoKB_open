# author : bwlee@kbfg.com

def count(arr1, ret = None):
    """
    gives count of each element in dictionary
    string is not affected    
    """
    if not ret:
        ret = {}
    for x in arr1:
        try:
            _ = iter(x)
            if type(x) is str:
                raise TypeError('do not iterate string')
            ret = count(x, ret)
        except TypeError:
            if x in ret:
                ret[x] += 1
            else:
                ret[x] = 1            
    return ret

def split(arr1, step):
    """ split a list with given step
    :param arr1: sequence like list
    :param step: size of each element

    arr1 = list( range(30) )
    for x in split(arr1, 3):
        print(x)

    ret = list( split(arr1, 3) )
    print(ret)

    ret2 = split(arr1, 3) 
    print(ret2)
    print(next(ret2))
    """
    for i in range(0, len(arr1), int(step)):
        yield arr1[i:i + step]

def isiterable(arr1):
    import collections
    """ check arr1 is iterable
    When arr1 is string or dict, do not treat them as iterable.
    Used in flatten
    """
    flag = not isinstance(arr1, str) and not isinstance(arr1, dict) and \
           isinstance(arr1, collections.Iterable)
    return flag

def flatten(arr1, depth = 1, level = 0, ret_list = None):
    """ Flatten iterable objects.

    :param arr1: objects to flatten
    :param depth: not used by user. It indicates applying depth
    if it is less than zero, flattens whole elements
    :param level: not for user, recursion parameter
    :param ret_list: not for user, recursion parameter

    :returns: result by its origin class type

    arr2 = ((1,3,4), (2,4,5), ((31,41,51), (12,15,16)), ((('a', 'b'), ('f','g')), ('d', 'e')))

    print(flatten(arr2))
    print(flatten(arr2, 2))
    """
    if level == 0:
        ret_list = []

    if depth < 0 :
        depth = 10000
    if level > depth:
        ret_list.append(arr1)
        return

    for x in arr1:
        if isiterable(x):
            flatten(x, depth, level+1, ret_list)
        else:
            ret_list.append(x)
    return type(arr1)(ret_list)

if __name__ == "__main__":
    '''
    xs = [1,1,3,54,65,3,4,5,6,2,3,4,1,1,4,5,43435,65,6,4,3]
    ys = [[1,1,3],[54,65,3,4],[5,6,2,3,4],1,1,4,[5,[43435],65,6],4,3, 'srt']
    print(count(xs))
    '''
    '''
    arr1 = list( range(30) )

    for x in split(arr1, 3):
        print(x)

    ret = list( split(arr1, 3) )
    print(ret)

    ret2 = split(arr1, 3) 
    print(ret2)
    print(next(ret2))
    '''

    arr1 = [[1,3,4], [2,4,5], [[31,41,51], [12,15,16]], [[['a', 'b'], ['f','g']], ['d', 'e']]]
    arr2 = ((1,3,4), (2,4,5), ((31,41,51), (12,15,16)), ((('a', 'b'), ('f','g')), ('d', 'e')))
    #arr2 = ((1,3,4), (2,4,5), ((31,41,51), (12,15,16)))

    #ret1 = flatten(arr1)
    #ret2 = flatten(arr2)
    ret1 = flatten(arr1, 1)
    ret2 = flatten(arr1, 2)
    ret3 = flatten(arr2, 1)
    ret4 = flatten(arr2, 2)
    print(ret1)
    print(ret2)
    print(ret3)
    print(ret4)
    print(flatten(arr1, 3))
    print(flatten(arr2, 3))


