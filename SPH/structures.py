class structure(object):
    '''
    Determines which energy groups from the fine structure belong to the coarse groups
    '''

    def __init__(self, fine_group, dgmstructure=None, order=None):
        if fine_group > 1:
            with open('XS/{}gXS.anlxs'.format(fine_group)) as f:
                f.readline()
                fine_bounds = [float(s) for s in f.readline().split()]
        else:
            fine_bounds = [2e7, 1e-11]

        if dgmstructure is not None:
            struct = self.change_structure(dgmstructure.structure, order)

            with open('XS/{}gXS.anlxs'.format(fine_group)) as f:
                f.readline()
                bounds = [float(s) for s in f.readline().split()]
            coarse_bounds = [fine_bounds[0]]
            test = 0
            for g, cg in enumerate(struct):
                if cg == test:
                    continue
                else:
                    test += 1
                    coarse_bounds.append(fine_bounds[g])
            coarse_bounds.append(fine_bounds[-1])
        else:
            coarse_bounds = [fine_bounds[0], fine_bounds[-1]]

        cg = 1
        grouping = []
        bounds = [fine_bounds[0]]
        for i, fg in enumerate(fine_bounds[1:]):
            if fg < coarse_bounds[cg]:
                cg += 1
                bounds.append(fine_bounds[i])

            grouping.append(cg)
        bounds.append(fine_bounds[-1])

        self.grouping = grouping
        self.bounds = bounds
        self.dE_fine = [fine_bounds[i] - fine_bounds[i+1] for i in range(len(fine_bounds)-1)]
        self.dE_coarse = [bounds[i] - bounds[i+1] for i in range(len(bounds)-1)]
        self.fine_bounds = fine_bounds
        self.coarse_bounds = coarse_bounds
        self.nFG = fine_group
        self.nCG = len(coarse_bounds) - 1

    def change_structure(self, structure, order):
        if order is not None:
            counts = {i: 0 for i in range(max(structure) + 1)}
            for s in structure:
                counts[s] += 1
            G = 0
            S = []
            oo = 0
            for i, c in counts.items():
                oo += min(order + 1, c)
                L = []
                nO = max((c + 1) // (order + 1), 1)
                nE = c % nO

                for j in range(min(order + 1, c)):
                    L += [G for _ in range(nO + (1 if j < nE else 0))]
                    G += 1
                S += L[:c]
            structure = S

        return structure


if __name__ == '__main__':
    class tempclass: pass

    o = tempclass()

    o.structure = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3]

    for i in range(26):
        S = structure(44, o, i)

        print(S.bounds)
        print(S.grouping)
