echo before filtering

echo --------------------

wc -l ./data/en-fa.txt/MIZAN.en-fa.en
wc -l ./data/en-fa.txt/MIZAN.en-fa.fa

echo --------------------

head ./data/en-fa.txt/MIZAN.en-fa.en -n 3
head ./data/en-fa.txt/MIZAN.en-fa.fa -n 3

echo --------------------
echo after filtering
echo --------------------

wc -l ./filtered_data/train.fa
wc -l ./filtered_data/train.en

echo --------------------

head ./filtered_data/train.fa -n 3
head ./filtered_data/train.en -n 3
