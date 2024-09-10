from model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

batch_size = 128
Max_epochs = 1000



def main(data_dir, model_dir, device):

    # get raw time-series data of training traffic data
    # train_data_be = np.load(os.path.join(data_dir, 'be.npy'))
    # train_data_ma = np.load(os.path.join(data_dir, 'ma.npy'))
    # data_be = np.load(os.path.join(data_dir, 'be.npy'))
    # ben = np.load(os.path.join(data_dir, 'BEN.npy'))
    rat = np.load(os.path.join(data_dir, 'RAT.npy'))
    pst = np.load(os.path.join(data_dir, 'PST.npy'))
    spt = np.load(os.path.join(data_dir, 'SPT.npy'))

    # ben_len = int(ben.shape[0] * 0.8)
    rat_len = int(rat.shape[0] * 0.8)
    pst_len = int(pst.shape[0] * 0.8)
    spt_len = int(spt.shape[0] * 0.8)

    # print(ben_len)
    print(rat_len)
    print(pst_len)
    print(spt_len)
    # data_ma = np.load(os.path.join(data_dir, 'ma.npy'))
    # other = np.load(os.path.join(data_dir, 'other.npy'))
    # print(train_data_ma.shape)
    # print(train_data_be.shape)
    # train_data = np.concatenate([data_be[:250, :50], ben[:250, :50], rat[: rat_len, :50], pst[:pst_len, :50], spt[:spt_len, :50]], axis=0)
    train_data = np.concatenate(
        [rat[: rat_len, :50], pst[:pst_len, :50], spt[:spt_len, :50]], axis=0)
    # test_ma = np.concatenate([ben[ben_len:, :], rat[rat_len:, :], pst[pst_len:, :], spt[spt_len:, :]], axis=0)
    # np.save()
    # print(train_data_ma.shape)
    # print(train_data_be.shape)
    # train_data = np.concatenate([train_data_be[:, :50], train_data_ma[:, :50]], axis=0)

    #
    # test_data = np.load(os.path.join(data_dir, 'test.npy'))
    # train_data = np.concatenate([train_data, test_data[:, :50]], axis=0)
    #train_data = np.load(os.path.join(data_dir, 'all.npy'))[:, :50]
    #print(train_data.shape)

    np.random.shuffle(train_data)
    total_size, input_size = train_data.shape
    print(total_size)
    device_id = int(device)
    print(device_id)
    torch.cuda.set_device(device_id)

    max_epochs = Max_epochs * 200 // total_size
    dagmm = LSTM_AE_GMM(
        input_size=input_size,
        max_len=2000,
        emb_dim=32,
        hidden_size=8,
        dropout=0.2,
        est_hidden_size=64,
        est_output_size=8,
        device=device_id,
    ).cuda()

    dagmm.train_mode()
    optimizer = torch.optim.Adam(dagmm.parameters(), lr=1e-2)
    for epoch in range(max_epochs):
        for batch in range(total_size // batch_size + 1):
            if batch * batch_size >= total_size:
                break
            optimizer.zero_grad()
            input = train_data[batch_size * batch : batch_size * (batch + 1)]
            loss = dagmm.loss(torch.Tensor(input).long().cuda())
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss)
        if (epoch + 1) % 50 == 0: # modified
            dagmm.to_cpu()
            dagmm = dagmm.cpu()
            torch.save(dagmm, os.path.join(model_dir, 'gru_ae_ma.pkl'))
            dagmm.to_cuda(device_id)
            dagmm = dagmm.cuda()

if __name__ == '__main__':
    data_dir = '../data/data'
    model_dir = '../pretrained'
    device = 0
    main(data_dir, model_dir, device)