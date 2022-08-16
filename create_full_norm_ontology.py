import shutil

dataset = 'ANATOMY'
folder = f'data/{dataset}/EmELpp'
out_file = f'data/{dataset}/inferences/{dataset}_norm_full.owl'

shutil.copyfile(f'{folder}/{dataset}_latest_norm_mod.owl', out_file)
with open(f'{folder}/{dataset}_test.txt', 'r') as infile:
    with open(out_file, 'a') as outfile:
        for line in infile:
            first, second = line.strip().split(' ')
            outfile.write(f'SubClassOf({first} {second})\n')
