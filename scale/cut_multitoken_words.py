import sys

def main():
    header = sys.stdin.readline()
    sys.stdout.write(header)
    dim = int(header.split()[1])
    for line in sys.stdin:
        fields = line.split()
        if len(fields) == dim + 1:  # includes word
            sys.stdout.write(line)


if __name__ == '__main__':
    main()
