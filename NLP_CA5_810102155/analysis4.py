import random


random.seed(42)

with open('./data/en-fa.txt/MIZAN.en-fa.en', 'r', encoding='utf-8') as en_file:
    en_lines = en_file.readlines()

with open('./data/en-fa.txt/MIZAN.en-fa.fa', 'r', encoding='utf-8') as fa_file:
    fa_lines = fa_file.readlines()

# Shuffle the lines in both datasets
zipped_data = list(zip(en_lines, fa_lines))
random.shuffle(zipped_data)
en_lines, fa_lines = zip(*zipped_data)

# Split the shuffled data into training, validation, and test sets
train_en = en_lines[:500000]
train_fa = fa_lines[:500000]

valid_en = en_lines[500000:505000]
valid_fa = fa_lines[500000:505000]

test_en = en_lines[505000:515000]
test_fa = fa_lines[505000:515000]

# Write the shuffled and split data to new files
def write_to_file(lines, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line)

write_to_file(train_en, './final_data/train.en')
write_to_file(train_fa, './final_data/train.fa')

write_to_file(valid_en, './final_data/valid.en')
write_to_file(valid_fa, './final_data/valid.fa')

write_to_file(test_en, './final_data/test.en')
write_to_file(test_fa, './final_data/test.fa')
