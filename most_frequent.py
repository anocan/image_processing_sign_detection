from collections import Counter
mylist = ['A', 'A', 25, 20, 30]
#print(max(k for k,v in Counter(mylist).items() if v>1))

accepted_first = {'A','B','C','D'}
accepted_second = {'1','2','3','4'}
accepted_third = {'X','Y','Z','T'}
accepted_fourth = {'1','2','3'}

i = ['A',2,'C','3']
if i[1] not in accepted_second:
    print(0)
else:
    print(1)