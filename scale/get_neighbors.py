import argparse
import pymagnitude as mag
import sys

def show(word, neighbors, count):
    for (tgt,score) in neighbors[:count]:
        row = '\t'.join([word, tgt, str(score)])
        print(row)


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        'source',
        help='source language word vector file (.magnitude)',
    )
    args.add_argument(
        'target',
        help='target language word vector file (.magnitude)',
    )
    args.add_argument(
        '-n',
        dest='count',
        type=int,
        default=10,
        help='number of neighbors per word',
    )
    argv = args.parse_args()
    src = mag.Magnitude(argv.source)
    tgt = mag.Magnitude(argv.target)
    for word in sys.stdin:
        word = word.rstrip()
        v = src.query(word)
        neighbors = tgt.most_similar(v)
        show(word, neighbors, argv.count)


if __name__ == '__main__':
    main()
