import numpy as np    


"""
Black-box functions for optimization.
All functions are scaled to have a minimum of 0.
"""

def beal(x):
    """
    Two dimensional. Unimodal.
    Bounds: -4.5 <= xi <= 4.5 
    """
    term1 = (1.5 - x[0] + x[0] * x[1])**2
    term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
    term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
    return term1 + term2 + term3


def himm(x):
    """
    Two dimensional. Multimodal.
    Bounds: -5 <= xi <= 5
    """
    x = np.array(x)
    return (((x[0]**2+x[1]-11)**2) + (((x[0]+x[1]**2-7)**2)))
    

def mccorm(x):
    """
    Two dimensional. Multimodal.
    Bounds: -1.5 <= x1 <= 4; -3 <= x2 <= 4
    """
    x = np.array(x)
    term1 = np.sin(x[0] + x[1])
    term2 = (x[0] - x[1])**2
    term3 = -1.5*x[0]
    term4 = 2.5*x[1] + 1
    return (term1 + term2 + term3 + term4) + 1.9133
    
    
def gold(x):
    """
    Two dimensional. Multimodal.
    Bounds: -2 <= xi <= 2
    """
    part1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    part2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return (part1 * part2) - 3


def camel6(x):
    """
    Two dimensional. Multimodal.
    Bounds: -3 <= x1 <= 3; -2 <= x2 <= 2
    """
    x = np.array(x)
    term1 = (4 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3) * x[0] ** 2
    term2 = x[0] * x[1]
    term3 = (-4 + 4 * x[1] ** 2) * x[1] ** 2
    return (term1 + term2 + term3) + 1.0316


def rosen(x):
    """
    N dimensional. Unimodal.
    Bounds: -30 <= xi <= 30
    """
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def spher(x):
    """
    N dimensional. Unimodal.
    Bounds: 0 <= xi <= 10
    """
    x = np.array(x)
    return np.sum(x ** 2)


def styb(x):
    """
    N dimensional. Multimodal.
    Bounds: -5 <= xi <= 5
    """
    x = np.array(x)
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x) + 39.16599*len(x)


def rast(x):
    """
    N dimensional. Multimodal.
    Bounds: -5.12 <= xi <= 5.12
    """
    n = len(x)
    x = np.array(x)
    return (10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def schwe(x):
    """
    N dimensional. Multimodal.
    Bounds: -500 <= xi <= 500
    """
    n = len(x)
    x = np.array(x)
    return 418.9829*n - np.sum(x * np.sin(np.sqrt(np.abs(x)))) 


def michal(x):
    """
    N dimensional. Multimodal.
    Bounds: 0 <= xi <= pi
    """
    n = len(x)
    assert n in [2, 5, 10]
    x = np.array(x)
    m = 10
    i = np.arange(1, len(x) + 1)  
    term = np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m)
    if n == 2: c = 1.8013
    elif n == 5: c = 4.687658
    elif n == 10: c = 9.66015
    return -np.sum(term) + c


def levy(x):
    """
    N dimensional. Multimodal
    Bounds: -10 <= xi <= 10
    """
    x = np.array(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    wi = w[:-1]
    sum_terms = np.sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2))
    
    return term1 + sum_terms + term3


def problem_all(p):

    problem_map = {
        "beal": {"func": beal, "dim": 2, "bounds": [(-4.5, 4.5)]},
        "himm": {"func": himm, "dim": 2, "bounds": [(-5, 5)]},
        "mccorm": {"func": mccorm, "dim": 2, "bounds": [(-1.5, 4), (-3, 4)]},
        "gold": {"func": gold, "dim": 2, "bounds": [(-2, 2)]},
        "camel6": {"func": camel6, "dim": 2, "bounds": [(-3, 3), (-2, 2)]},
        "rosen": {"func": rosen, "dim": None, "bounds": [(-30, 30)]},
        "spher": {"func": spher, "dim": None, "bounds": [(0, 10)]},
        "styb": {"func": styb, "dim": None, "bounds": [(-5, 5)]},
        "rast": {"func": rast, "dim": None, "bounds": [(-5.12, 5.12)]},
        "schwe": {"func": schwe, "dim": None, "bounds": [(-500, 500)]},
        "michal": {"func": michal, "dim": None, "bounds": [(0, np.pi)]},
        "levy": {"func": levy, "dim": None, "bounds": [(-10, 10)]}}
    
    return problem_map[p]


def truth(p):
    f = problem_all(p)
    if len(f["bounds"]) == 2:
        x1 = np.linspace(f["bounds"][0][0], f["bounds"][0][1], 500)
        x2 = np.linspace(f["bounds"][1][0], f["bounds"][1][1], 500)
    else:
        x1 = np.linspace(f["bounds"][0][0], f["bounds"][0][1], 500)
        x2 = np.linspace(f["bounds"][0][0], f["bounds"][0][1], 500)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f["func"]([X1[i, j], X2[i, j]])
    return X1, X2, Z