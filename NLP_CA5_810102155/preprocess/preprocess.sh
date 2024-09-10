fairseq-preprocess --source-lang en --target-lang fa \
--trainpref ../final_data/train/train \
--validpref ../final_data/valid/valid \
--testpref ../final_data/test/test \
--destdir ./data_bin/ --nwordstgt 10000 --nwordssrc 10000 \
--workers 20
