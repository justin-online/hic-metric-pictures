import sys

with open(sys.argv[1]) as infile:
    count = 0
    with open(sys.argv[2], 'a') as outfile:
        for line0 in infile:
            line = line0.strip()
            if count == 0:
                outfile.write(line + '\n')
            else:
                if int(line.split()[13]) > 1:
                    outfile.write(line + '\n')
            count += 1
