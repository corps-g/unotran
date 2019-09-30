class structure(object):
    '''
    Determines which energy groups from the fine structure belong to the coarse groups
    '''

    def __init__(self, fine_group, dgmstructure=None):
        if fine_group > 1:
            with open('XS/{}gXS.anlxs'.format(fine_group)) as f:
                f.readline()
                fine_bounds = [float(s) for s in f.readline().split()]
        else:
            fine_bounds = [2e7, 1e-11]

        if dgmstructure is not None:
            with open('XS/{}gXS.anlxs'.format(fine_group)) as f:
                f.readline()
                bounds = [float(s) for s in f.readline().split()]
            coarse_bounds = [fine_bounds[0]]
            test = 0
            for g, cg in enumerate(dgmstructure.structure):
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


if __name__ == '__main__':
    S = structure(7, 7)
    print(S.bounds)
    print(S.grouping)
