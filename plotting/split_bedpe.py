import sys

n = int(sys.argv[2])

writers = []
for k in range(n):
    f = open(sys.argv[1] + '_' + str(k) + '.bedpe', 'w')
    writers.append(f)

with open(sys.argv[1]) as f:
    for index, line in enumerate(f):
        f_index = index % n
        writers[f_index].write(line.strip() + '\n')

for writer in writers:
    writer.close()
