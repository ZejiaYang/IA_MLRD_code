from tick6 import *
import random
'''
1. Random guesser: knows the overall distribution of the categories and chooses between them according to that proportion.
2. Happy random guesser: chooses positive 60% of the time, neutral 20%, negative 20%.
3. Doesn’t sit on the fence: chooses positive 50% of the time, negative 50% of the time.
4. Middle of the road: chooses neutral 80% of the time.
Code some number of random agents and let them make choices for 50 examples and calculate kappa. Repeat this exercise 100 times.
'''
random.seed(42)
def random_guesser(n:int, m:int) -> List[Dict[int, int]]:
    # Assume the overal distribution of postive is 0.35, negative is 0.35 and neutral is 0.3
    tguess = []
    for _ in range(m):
        guess = {}
        for i in range(n):
            d = random.uniform(0, 1)
            if d < 0.35:
                a = 1
            elif d < 0.7:
                a = -1
            else:
                a = 0
            guess[i] = a
        tguess.append(guess)
    return tguess

def happy_guesser(n:int, m:int) -> List[Dict[int, int]]:
    hguess = []
    for _ in range(m):
        guess = {}
        for i in range(n):
            d = random.uniform(0, 1)
            if d < 0.6:
                a = 1
            elif d < 0.8:
                a = -1
            else:
                a = 0
            guess[i] = a
        hguess.append(guess)
    return hguess

def half_guesser(n:int, m:int) -> List[Dict[int, int]]:
    fguess = []
    for _ in range(m):
        guess = {}
        for i in range(n):
            d = random.uniform(0, 1)
            if d <= 0.5:
                a = 1
            else:
                a = -1
            guess[i] = a
        fguess.append(guess)
    return fguess

def middle_guesser(n:int, m:int) -> List[Dict[int, int]]:
    mguess = []
    for _ in range(m):
        guess = {}
        for i in range(n):
            d = random.uniform(0, 1)
            if d <= 0.8:
                a = 0
            elif d <= 0.9:
                a = 1
            else:
                a = -1
            guess[i] = a
        mguess.append(guess)
    return mguess

if __name__ == '__main__':
    guesser = [random_guesser, happy_guesser, half_guesser, middle_guesser] 
    names = ['random_guesser', 'happy_guesser', 'half_guesser', 'middle_guesser']
    kappas = {}
    n = 50
    for i in range(100):
        for g1, n1 in zip(guesser, names):
            for g2, n2 in zip(guesser, names):
                pred1 = g1(n, 50)
                pred2 = g2(n, 50)
                pred = pred1 + pred2
                agreement_table = get_agreement_table(pred)
                kappa = calculate_kappa(agreement_table)
                na = n1 + " " + n2
                if na not in kappas.keys():
                    kappas[na] = kappa
                else:
                    kappas[na] += kappa
    
    for names, kappa in kappas.items():
        n1, n2 = names.split()
        print(f"kappa for {n1} & {n2} is {kappa/100}")
