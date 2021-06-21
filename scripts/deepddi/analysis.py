import numpy as np
import matplotlib.pyplot as plt


def main():
    with open('data/deepddi/KnownDDI.csv') as deepddi:
        lines = deepddi.readlines()
        lines = [a.split(',') for a in lines]
    node_degrees = {}
    edge_freq = {}
    for trip in lines:
        n1, n2, rel = trip
        if n1 in node_degrees:
            node_degrees[n1] += 1
        else:
            node_degrees[n1] = 1
        if n2 in node_degrees:
            node_degrees[n2] += 1
        else:
            node_degrees[n2] = 1
        if rel in edge_freq:
            edge_freq[rel] += 1
        else:
            edge_freq[rel] = 1

    degrees = np.array(list(node_degrees.values()))
    degrees.sort()
    edge_freq = np.array(list(edge_freq.values()))
    edge_freq.sort()

    print(degrees.shape[0])

    # this may kill process on small computers
    plt.plot(range(degrees.shape[0]), degrees)
    plt.show()

    print(edge_freq.min())

    plt.bar(range(edge_freq.shape[0]), edge_freq)
    plt.title('Deepddi Imalance')
    plt.xlabel('Relation')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    main()
