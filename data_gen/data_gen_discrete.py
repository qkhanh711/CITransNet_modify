import os
import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=2, help='Number of bidders')
parser.add_argument('--m', type=int, default=5, help='Number of items')
parser.add_argument('--n_type', type=int, default=10, help='Number of bidder types')
parser.add_argument('--m_type', type=int, default=10, help='Number of item types')

args = parser.parse_args()
bidders = args.n
items = args.m
n_type = args.n_type
m_type = args.m_type

def gen(n, m, n_type=10, m_type=10, phase='train', n_data=200000):
    dir = f'../data_multi/{n_type}t{m_type}t_{n}x{m}/{phase}_{n_data}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    raw_data = np.zeros((n_data, n, n_type, m, m_type))
    ind = [(i, j) for i in range(n_type) for j in range(m_type)]
    for i,j in tqdm(ind):
        myclip_a, myclip_b, my_mean, my_std = 0, 1, (1 + (i+j+2)%10)/11, 0.05
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        x = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=(n_data, n, m))
        raw_data[:, :, i, :, j] = x
    tau = np.random.randint(0, n_type, size=(n_data, n))
    b_data = raw_data.reshape(n_data*n, n_type, m, m_type)
    b_data = b_data[np.arange(n_data*n), tau.reshape(-1)]
    b_data = b_data.reshape(n_data, n, m, m_type)

    omega = np.random.randint(0, m_type, size=(n_data, m))
    data = b_data.transpose(0, 2, 3, 1) # n_data, m, m_type, n
    data = data.reshape(n_data*m, m_type, n)
    data = data[np.arange(n_data*m), omega.reshape(-1)]
    data = data.reshape(n_data, m, n).transpose(0, 2, 1)

    valuation = data
    np.save(os.path.join(dir, 'trueValuations'), valuation)
    np.save(os.path.join(dir, 'Agent_names_idxs'), tau)
    np.save(os.path.join(dir, 'Object_names_idxs'), omega)

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    from typing import List
    if items != 3:
        gen(2, 5, n_type, m_type, phase='test', n_data=5000)
        gen(2, 5, n_type, m_type, phase='training', n_data=int(1e5))
        gen(3, 10, phase='test', n_data=5000)
        gen(3, 10, phase='training', n_data=int(2e5))
        gen(5, 10, phase='test', n_data=5000)
        gen(5, 10, phase='training', n_data=int(2e5))
    gen(n=bidders, m=items, phase='test', n_data=5000)
    gen(n=bidders, m=items, phase='training', n_data=int(1e5))
