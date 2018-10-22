
def readData(matID, g, f):
    '''
    Read the data for a given material from the file
    '''
    # Define the line indices in the file
    begin = 3 + matID * (2 + G * 9)
    end = begin + (1 + G * 9) + 1

    return f[begin:end]

def formDataset(g):
    names = ['uo2', 'moxlow', 'moxmid', 'moxhigh', 'BWR1', 'BWR2', 'BWR3']
    morenames = ['water']

    # Write the header for the XS file
    header = '{} {} 0\n'.format(len(names) + len(morenames), g)

    mats = []
    moreMats = []
    for i, name in enumerate(names):
        # Open and read the cross sections file for the current name
        f = open('{}g/{}-{}.inp.anlxs'.format(g, name, g), 'r').readlines()

        # If this is the first material
        if i == 0:
            # Write the velicity and energy bounds to the header
            header += f[1]
            header += f[2]

        # Read the fuel from the file
        fuelData = readData(1, g, f)

        # Replace the matID number with the name
        fuelData[0] = '{}\n'.format(name)

        # Write the cross section data to the string
        mats.append(''.join(fuelData))

        # If this is the first material
        if i == 0:
            # Read the water from the file
            waterdata = readData(2, g, f)

            # Replace the matID number with the name
            waterdata[0] = 'water\n'

            # Write the cross section data to the string
            moreMats.append(''.join(waterdata))

    # Write the strings to the file
    with open('{}g/{}gXS.anlxs'.format(g, g), 'w') as f:
        f.write(header)
        for s in mats + moreMats:
            f.write(s)

if __name__ == '__main__':
    Gs = [2, 44, 238]
    for G in Gs:
        formDataset(G)


