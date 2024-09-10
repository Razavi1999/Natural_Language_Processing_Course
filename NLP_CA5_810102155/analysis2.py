import collections
import numpy as np
import random
from matplotlib import pyplot as plt


def tokenize_plot_histogram(path , filename):
    filename = './histograms/' + filename

    token_count = []

    file = open(path, 'r')
    Lines = file.readlines()

    for line in Lines:
        count = len(line.split())
        token_count.append(count)

    bins = np.arange(1, 50, 2)
    plt.xlim([min(token_count) - 5, max(token_count) + 5])

    plt.hist(token_count, bins=bins, alpha=0.5)
    plt.title('token counts ' + filename)
    plt.xlabel('counts of tokens')
    plt.ylabel('frequency of counts of tokens')

    plt.savefig(filename)
    plt.show()


def reduce_volume(fa_path, en_path, fa_filter_path, en_filter_path):
    file = open(fa_path , 'r')
    Lines = file.readlines()

    file2 = open(en_path , 'r')
    Lines2 = file2.readlines()

    lines_to_write = []

    index = 0
    for line in Lines:
        count = len(line.split())
        if 10 <= count <= 50:
            lines_to_write.append(index)

        index = index + 1

    file_fa_filter_path = open(fa_filter_path , 'w')
    file_en_filter_path = open(en_filter_path , 'w')

    for item in lines_to_write:
        file_fa_filter_path.write(Lines[item])
        file_en_filter_path.write(Lines2[item])
#
#
#
# tokenize_plot_histogram(
#     path='./data/en-fa.txt/MIZAN.en-fa.fa' ,
#     filename = 'train_fa_histogram'
# )
#
# tokenize_plot_histogram(
#     path='./data/en-fa.txt/MIZAN.en-fa.en' ,
#     filename = 'train_en_histogram'
# )


# reduce_volume(
# fa_path = './data/en-fa.txt/MIZAN.en-fa.fa' ,
# en_path = './data/en-fa.txt/MIZAN.en-fa.en' ,
# fa_filter_path = './filtered_data/train.fa' ,
# en_filter_path = './filtered_data/train.en'
#               )
