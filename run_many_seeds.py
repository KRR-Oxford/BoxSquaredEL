import gc

import torch
import numpy as np
from train import run
import time
from datetime import timedelta


def main():
    # the following seeds were randomly chosen by executing
    # seeds = [(random.randint(0, 10**5), random.randint(0, 10**5)) for _ in range(10)]
    seeds = [(38588, 53121), (52065, 26435), (47121, 66163), (21683, 91177), (3206, 51103), (43180, 2475),
             (32510, 3548), (79126, 75212), (34641, 40480), (87167, 7729)]

    num_seeds = 5

    start = time.time()
    all_rankings = []
    for seed in seeds[:num_seeds]:
        torch.manual_seed(seed[0])
        np.random.seed(seed[1])
        rankings = run(use_wandb=False)
        all_rankings.append(rankings)
        torch.cuda.empty_cache()
        gc.collect()
    end = time.time()
    print(f'Took {str(timedelta(seconds=int(end - start)))}s.')

    results = average_rankings(all_rankings)
    output = ''
    csv_output = ''
    for tup in results:
        output += f'top1: {tup[0]:.2f}, top10: {tup[1]:.2f}, ' \
                  f'top100: {tup[2]:.2f}, mean: {tup[5]}, median: {tup[3]}, ' \
                  f'mrr: {tup[4]:.2f}, auc: {tup[6]:.2f}\n\n'

        csv_output += f'{tup[0]:.2f},{tup[1]:.2f},' \
                      f'{tup[2]:.2f},{tup[3]},{tup[4]:.2f},' \
                      f'{tup[5]},{tup[6]:.2f}\n'

    print('\n')
    print(output)
    with open('avg_output.txt', 'w+') as f:
        f.write(output)
    with open('avg_output_csv.txt', 'w+') as f:
        f.write(csv_output)


def average_rankings(all_rankings):
    results = []
    for tup in zip(*all_rankings):
        top1 = np.mean([r.top1 / len(r) for r in tup])
        top10 = np.mean([r.top10 / len(r) for r in tup])
        top100 = np.mean([r.top100 / len(r) for r in tup])
        median = round(np.mean([np.median(r.ranks) for r in tup]))
        mrr = np.mean([np.mean([1 / x for x in r.ranks]) for r in tup])
        mean = round(np.mean([np.mean(r.ranks) for r in tup]))
        auc = np.mean([r.auc for r in tup])
        results.append((top1, top10, top100, median, mrr, mean, auc))

    return results


if __name__ == '__main__':
    main()
