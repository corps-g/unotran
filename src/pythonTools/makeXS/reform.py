
def formDataset(g):
    names = ['uo2', 'moxlow', 'moxmid', 'moxhigh']
    morenames = ['water', 'void']

    header = '6 {} 0\n'.format(g)
    mats = []
    for i, name in enumerate(names):
        f = open('{}g/{}-{}.inp.anlxs'.format(g, name, g), 'r').readlines()
        if i == 0:
            header += f[1]
            header += f[2]

        s = '{}\n'.format(name)
        for l in f[4:5 + g * 9]:
            s += l

        mats.append(s)

        if i == 0:
            moreMats = []
            for m in range(2):
                s = morenames[m] + '\n'
                jump = (m + 1) * (m * 1 + g * 9)
                for l in f[jump + 6: jump + 7 + g * 9]:
                    s += l
                moreMats.append(s)

    #print moreMats[0]

    with open('{}g/{}gXS.anlxs'.format(g, g), 'w') as f:
        f.write(header)
        for s in mats:
            f.write(s)
        for s in moreMats:
            f.write(s)

if __name__ == '__main__':
    Gs = [1968]
    for G in Gs:
        formDataset(G)


