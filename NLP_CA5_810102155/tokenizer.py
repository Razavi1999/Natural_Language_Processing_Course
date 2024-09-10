import sentencepiece as spm

train_en_file = './final_data/train.en'
train_fa_file = './final_data/train.fa'

valid_en_file = './final_data/valid.en'
valid_fa_file = './final_data/valid.fa'

test_en_file = './final_data/test.en'
test_fa_file = './final_data/test.fa'

tokenizer_dir = './tokenizer'


spm.SentencePieceTrainer.train(input=train_en_file, \
                               model_prefix = f'{tokenizer_dir}/train/en', \
                               vocab_size=10000, model_type='bpe')


spm.SentencePieceTrainer.train(input=train_fa_file, \
                               model_prefix=f'{tokenizer_dir}/train/fa', \
                               vocab_size=10000, model_type='bpe')

spm.SentencePieceTrainer.train(input = valid_en_file , \
                               model_prefix = f'{tokenizer_dir}/valid/en' , \
                               vocab_size = 10000 , model_type = 'bpe'
                               )

spm.SentencePieceTrainer.train(input = valid_fa_file , \
                               model_prefix = f'{tokenizer_dir}/valid/fa' , \
                               vocab_size = 10000 , model_type = 'bpe'
                               )

spm.SentencePieceTrainer.train(input = test_en_file , \
                               model_prefix = f'{tokenizer_dir}/test/en' , \
                               vocab_size = 10000 , model_type = 'bpe'
                               )

spm.SentencePieceTrainer.train(input = test_fa_file , \
                               model_prefix = f'{tokenizer_dir}/test/fa' , \
                               vocab_size = 10000 , model_type = 'bpe'
                               )


