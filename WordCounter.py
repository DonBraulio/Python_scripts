"""
Auto assessment for Udacity's Self Driving Car course.
Count the number of times each word appears in the string "s".
Return array with the first "n" tuples (word, freq),
ordering by most frequent words, and alphabetically in case of tie."""

from operator import itemgetter


def count_words(s, n):
    counted = set()
    counter = []
    for word in s.split(' '):
        if word not in counted:
            counted.add(word)
            counter.append((word, 1))
        else:
            idx = next(i for i, x in enumerate(counter) if x[0] == word)
            counter[idx] = (word, counter[idx][1] + 1)
    result = sorted(sorted(counter, key=itemgetter(0)), key=itemgetter(1), reverse=True)
    return result[0:n]
