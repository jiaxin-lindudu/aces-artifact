
def average_statistics(stat_list):
    S = sum(S for S, n in stat_list)
    n = sum(n for S, n in stat_list)
    return S, n
