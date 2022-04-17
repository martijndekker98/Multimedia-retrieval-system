from math import comb


# Computes the probability of getting X from a sample, with querysize qsize.
# In terms of a dice: probability of throwing X eyes, given qsize throws.
def compProbabilityX(x: int, qsize: int, inClass: int = 19, inTotal: int = 379):
    prob = 1.0
    for i in range(0, x):
        prob *= (inClass-i)/(inTotal-i)
        # print(f"{inClass-i} / {inTotal-i} = {(inClass-i)/(inTotal-i)}")
    for i in range(0, (qsize -x)):
        prob *= (inTotal - inClass - i)/(inTotal-i-x)
        # print(f"{inTotal - inClass - i} / {inTotal-i-x} = {(inTotal - inClass - i)/(inTotal-i-x)}")
    ncr = comb(qsize,x)
    # print(f"NCR: {ncr}")
    return ncr*prob


# computes the expected number of TP (models) for a query size qsize
def compProbabilityQ(qsize: int):
    gemiddeld = 0
    for aantal in range(0, qsize+1):
        a = compProbabilityX(aantal, qsize)
        # print(f"De kans is {a}, aantal: {aantal*a}")
        gemiddeld += aantal*a
    # print(gemiddeld)
    return gemiddeld

# Compute the predicted TPs, used by ROC computations
def compROCValues():
    specificities = [1.0]
    recalls = [0.0]
    for qs in range(1, 380):
        TP = compProbabilityQ(qs)
        recalls.append( TP/19 )
        TN = 360 - (qs-TP)
        specificities.append( TN/360 )
    return recalls, specificities


# testing
def main():
    # print(compProbabilityX(1, 5))
    qsize = 40
    a = compProbabilityQ(qsize)
    TP = a
    FP = qsize - a
    FN = 19 - a
    TN = 379 - TP - FP - FN
    print(f"ACC: {(TP+TN) / (379)}")
    a, b = compROCValues()
    print(a)
    print(b)

# testing
def main2():
    qsizes = [5, 10, 20, 40]
    for q in qsizes:
        TP = compProbabilityQ(q)
        prec = TP / q
        recall = TP / 19
        f1 = (2*prec*recall) / (prec + recall)
        print(f"For q size {q}, prec: {prec}, recall: {recall}, f1: {f1}")


if __name__ == '__main__':
    main2()