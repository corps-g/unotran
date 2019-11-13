import os

G = 44

review = []
unstable = []
for fname in os.listdir():
    if not 'z_run_full{}'.format(G) in fname:
        continue
    print(fname)
    flag = False
    task = int(fname.split('_')[-1][:-4])

    with open(fname, 'r') as f:
        for line in f.readlines():
            if 'exiting' in line:
                unstable.append(task)
                break
            if 'complete' in line or 'stopping' in line:
                flag = True
        else:
            if flag:
                os.system('rm {}'.format(fname))
            else:
                review.append(task)
print('The following tasks are still running:')
print(sorted(list(set(review))))
print('The following tasks were unstable:')
print(sorted(list(set(unstable))))
