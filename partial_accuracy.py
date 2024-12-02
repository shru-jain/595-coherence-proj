
def compute_accuracy(pair_list):
    # binary_score, abs_dist_score
    length = len(pair_list)
    score = 0
    for a, b in pair_list:
        if a == b:
            score += 1
    return(float(score)/length)

def compute_partial_accuracy(pair_list):
    length = 0
    score = 0
    for a, b in pair_list:
        for i in range(len(a)):
            length += 1
            if a[i] == b[i]:
                score += 1
    return(float(score)/length)
    
