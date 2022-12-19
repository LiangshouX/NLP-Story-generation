"""
    加载数据集的功能类
"""

import argparse
import torch
import pandas as pd
import torch.utils.data
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, dataPath):
        self.args = args
        self.dataPath = dataPath
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        """加载数据集"""
        train_df = pd.read_csv(self.dataPath)
        # text1 = train_df['sentence1'].str.cat(sep=' ').split(' ')
        # text2 = train_df['sentence2'].str.cat(sep=' ').split(' ')
        # text3 = train_df['sentence3'].str.cat(sep=' ').split(' ')
        # text4 = train_df['sentence4'].str.cat(sep=' ').split(' ')
        # text5 = train_df['sentence5'].str.cat(sep=' ').split(' ')
        num_lines = train_df['sentence1'].shape[0]
        words = []
        for num_line in range(num_lines):
            words += train_df.iloc[num_line, 2:].str.cat(sep=" ").split(" ")
        return words

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=4)
    parser.add_argument('--saveStep', type=int, default=2)
    args = parser.parse_args()

    dataset = Dataset(args, 'data/train1.csv')

    words_list = dataset.load_words()
    print(len(words_list), '\n', words_list[:20])

    # for batch, (x, y) in enumerate(dataset):
    #     print(batch, x, y)
    #     break

    mytes = pd.read_csv("data/train1.csv", converters={'column_name': eval})
    # tex1 = pd.Series(mytes.iloc[:4, 2:])
    tex = mytes['sentence1'].shape
    print('text', tex)

    # num_lines = len(mytes['sentence1'].str.cat(sep=' ').split(" "))
    # print(num_lines)

    text1 = mytes.iloc[0, 2:].str.cat(sep=' ').split(" ")
    text2 = []
    for i in range(30000):
        text2 += mytes.iloc[i, 2:].str.cat(sep=' ').split(" ")

    print(len(text2), '\n', text2[:100])


