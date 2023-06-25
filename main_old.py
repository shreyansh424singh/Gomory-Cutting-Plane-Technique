import numpy as np

def simplex(c, A, b):
    m, n = A.shape
    basic_indices = np.arange(n-m, n)
    tableau = np.zeros((m+1, n+1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b
    tableau[-1,:-1] = c
        
    while True:
        # Check if optimal solution is found
        if np.all(tableau[-1, :-1] >= 0):
            break   
        # Select pivot column
        j=0
        for k in range(n) :
            if tableau[-1, k] < 0:
                j=k
                break
      
        if np.all(tableau[:-1, j] <= 0):
            break

        i=0
        mi = 100000000
        for k in range(m):
            if tableau[k,j] > 0:
                cmi = tableau[k,-1] / tableau[k,j]
                if(cmi < mi) :
                    i = k
                    mi = cmi
                
        
        # Perform pivot operation
        tableau[i] /= tableau[i, j]
        for k in range(m+1):
            if k != i:
                tableau[k] -= tableau[k, j] * tableau[i]
        basic_indices[i] = j

    # Extract solution and optimal objective value
    x = np.zeros(n)
    x[basic_indices] = tableau[:-1, -1]
    obj = tableau[-1, -1]
    A_obj = tableau[:-1, :-1]
    b_obj = tableau[:-1, -1]
    c_obj = tableau[-1,:-1]

    return x, obj, A_obj, b_obj, c_obj, basic_indices, x


def dual_simplex(c, A, b, cost, basic_indices):
    m, n = A.shape
    # Formulate initial tableau
    tableau = np.zeros((m+1, n+1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b
    tableau[-1, :-1] = c
    tableau[-1, -1] = cost

    while True:
        # Check if dual solution is feasible
        if np.all(tableau[:-1, -1] >= 0):
            break

        # Select leaving variable
        j=0
        for k in range(m) :
            if tableau[k, -1] < 0:
                j=k
                break
       
        if np.all(tableau[j, :-1] >= 0):
            return None
        
        i=0
        mi = 100000000
        for k in range(n):
            if tableau[j,k] < 0:
                cmi = tableau[-1,k] / abs(tableau[j,k])
                if(cmi < mi) :
                    i = k
                    mi = cmi
       

        # Perform pivot operation
        tableau[j] /= tableau[j, i]
        for k in range(m+1):
            if k != j:
                tableau[k] -= tableau[k, i] * tableau[j] 
        basic_indices[j] = i
        
    x = np.zeros(n)
    x1 = np.zeros(n)
    x1[basic_indices-n] = tableau[:-1, -1]
    x = tableau[:-1, -1]
    obj = tableau[-1, -1]
    A_obj = tableau[:-1, :-1]
    b_obj = tableau[:-1, -1]
    c_obj = tableau[-1,:-1]
    
    return x, obj, A_obj, b_obj, c_obj, x1

def gomory_cutting_plane(c, A, b):
    m, n = A.shape

    # Solve LP relaxation to get fractional solution
    lp_solution, obj, As, bs, cs, basic_indices, answer = simplex(c, A, b)

    while True:
        m, n = As.shape

        lp_solution_integers = np.round(lp_solution)

        # Check if solution is integer
        if np.all(np.abs(lp_solution_integers-lp_solution) < 1e-4) :
            return np.round(answer), round(obj)
        
        j = 0
        for j_i in range(len(bs)):
            if (np.abs(lp_solution_integers[j_i]-lp_solution[j_i]) >= 1e-4) : 
                j = j_i
                break

        basic_indices = np.append(basic_indices, n)
    
        b_n = np.zeros((1,))
        b_n[0] = -1*(bs[j] - np.floor(bs[j]))
        b_new = np.concatenate((bs,b_n))
        A_new = np.hstack((As, np.zeros((m,1))))
        ar_new = np.ones((n+1,))
        for i in range(n) :
            ar_new[i] = -1*( As[j][i] - np.floor(As[j][i]))
        A_new = np.vstack((A_new,ar_new))
        c_new = np.concatenate((cs,np.zeros((1,))))

        lp_solution, obj, As, bs, cs, answer = dual_simplex(c_new, A_new, b_new, obj, basic_indices)
        
        if obj == float('inf'): 
            return None


if __name__ == "__main__" : 
    with open("test7.txt", "r") as f:
        n, m = map(int, f.readline().split())
        b = list(map(int, f.readline().split()))
        c = list(map(int, f.readline().split()))
        A = []
        for i in range(m):
            row = list(map(int, f.readline().split()))
            A.append(row)

        A = np.array(A)
        ident = np.eye(m,m)
        A = np.hstack((A,ident))    
        b = np.array(b)
        c = np.array(c) * -1
        c = np.concatenate((c ,np.zeros(m,)))
       
    x, obj = gomory_cutting_plane(c, A, b)

    x = x[0:n]

    print("Final answer here")
    print(x)
    print(obj)