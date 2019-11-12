import os

s = '#!/bin/bash -l\n'
s += '\n#SBATCH --mem-per-cpu=1G\n'
s += '#SBATCH --time=1:00:00\n'
s += '#SBATCH --job-name=run_full44\n'
s += '#SBATCH --output=z_run_full44-%A_%a.out\n'
s += '#SBATCH --array={}\n'
s += '#SBATCH --mail-user=rlreed@ksu.edu --mail-type=FAIL,END\n'
s += 'echo $PWD\n'
s += 'python -u run_full.py\n'

items = items = [1118]
items = sorted(items)


array = []
for i, item in enumerate(items):
    array.append(item)
    try:
        if item + 1 == items[i + 1]:
            continue
    except IndexError:
        pass

    nums = '{}'.format(item) if len(array) == 1 else '{}-{}'.format(array[0], array[-1])

    print('submitting {}'.format(nums))
    with open('submit.sh', 'w') as f:
        f.write(s.format(nums))

    array = []
    os.system('sbatch submit.sh')
