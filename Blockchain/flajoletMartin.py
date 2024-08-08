def printTable(input, hash, binHash, trailingZeros): 
    print('\nCalculation\t\tHash\tBinary\tTrailing Zeros')
    print('-----------\t\t----\t------\t--------------')
    for i in range(len(input)):
        print(f'(3*({input[i]})+1) mod 5', end='\t\t')
        print(hash[i], end='\t')
        print(binHash[i], end='\t\t')
        print(trailingZeros[i], end='\n')

def flajoletMartinAlgorithm(input):
    # finding hash with hardcoded function
    hash = [(((3*x)+1)%5) for x in input]
    # converting hash values to binary
    binHash =  [format(x, '03b') for x in hash]
    # counting trailing zeros
        # the need of condition here is for all 0 case where the length remains same
    trailingZeros = [( len(x) - ( len(x.rstrip('0')) if len(x.rstrip('0'))!=0 else len(x) ) ) for x in binHash]
    printTable(input, hash, binHash, trailingZeros)
    maxZeros = max(trailingZeros)
    return 2 ** maxZeros

input = [int(i) for i in input('\nInput Stream: ').split()]
print('\nEstimated number of Distint Elements:', flajoletMartinAlgorithm(input))
print()