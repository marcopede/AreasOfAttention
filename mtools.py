## This file contains small utilities that I use for debugging.

def print_sent(x_sent, i):
    """ Given a set of sentences encoded with their number and an index for which sentence to pick, 
    maps back to words."""
    print(" ".join([index_to_word[vx_sent[6,i]] for i in range(21)][1:]).split('#END')[0])

