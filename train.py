import argparse
import csv
import os
import time
import torch
import numpy as np
import torch.nn.functional
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from torch import Tensor

def train(args):
    t1 = time.time()
    model.train()

    dataloader = DataLoader(
        data_train,
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        ##########
        ##########
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        if (epoch+1) % args.saveStep == 0:
            torch.save(model, 'models/chekpoint%d.pt' % (epoch + 1))

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                print("Training on %s======>" % str(device), {'epoch': epoch, 'batch': batch, 'loss': loss.item()},
                      "Time cost:%.6f s" % (time.time() - t1))

    print("Training Done!")


def predict(text, next_words=100):
    """"""


def predict_(dataset, model, text, next_words=100):
    words = text.split(' ')
    model.eval()

    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        x = x.to(device)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(Tensor.cpu(last_word_logits), dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=4)
    parser.add_argument('--saveStep', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_train = Dataset(args, dataPath='data/train1.csv')
    data_test = Dataset(args, dataPath='data/test1.csv')

    n_vocab = max(len(data_train.uniq_words), len(data_test.uniq_words))
    model = Model(n_vocab).to(device)

    # if not os.path.exists('models/chekpoint%d.pt' % args.max_epochs):
    train(args)

    text = "Tommy was very close to his dad and loved him greatly."
    # print(" ".join(predict(text)))

    print(" ".join(predict_(data_train, torch.load('models/chekpoint30.pt'), text)))

    # 将生成的内容写入csv
    with open('data/result.csv', 'w') as f:
        writer = csv.writer(f)

        csvReader = csv.reader(open('data/test1.csv'))
        for line in csvReader:
            if line[2] == 'sentence1':
                continue

            try:
                words = " ".join(predict_(data_train, torch.load('models/chekpoint30.pt'), line[2]))
                res = words.split('.')[:5]
                writer.writerow(res)
                # print(res)
            except KeyError:
                print("skip")
