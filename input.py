with open("input.txt", "r") as f:
    n, m = map(int, f.readline().split())
    b = list(map(int, f.readline().split()))
    c = list(map(int, f.readline().split()))
    A = []
    for i in range(m):
        row = list(map(int, f.readline().split()))
        A.append(row)
