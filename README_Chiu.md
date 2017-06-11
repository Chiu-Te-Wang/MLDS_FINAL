# Early Stop 規則在 Recurrent Neural Network 上是否有效果
Lutz Prechelt做了一連串的實驗，顯示各種 “Early stopping” 的方法在performance方面的確是有些許的幫助，然而卻得用大量的training time作為代價。 而這個結論是只有在他所做的fully connected NN上，還是在其他架構的 neural networks 上也是亦然呢? (Reference paper : Prechelt, Lutz. “Early stopping—but when?.” Neural Networks: Tricks of the Trade. Springer Berlin Heidelberg, 2012. 53-67.)
<br/>

##文章
[文章](https://ntumlds.wordpress.com/2017/03/25/r05922007_ai%e4%ba%ba%e5%b7%a5%e6%99%ba%e9%9a%9c/)

## Requirements
- Python3
- Numpy
- Tensorflow 1.0
- NLTK 3.2.1
- SciPy
- scikit-learn
- gensim

## Prepare data
### 下載 Training and Test data set:(已下載)
	https://drive.google.com/open?id=0B5eGOMdyHn2mWDYtQzlQeGNKa2s
	Please extract the training data and store them inside the ./data directory.
### 下載 Pretrained Model:
	wget -O model.tgz https://www.dropbox.com/s/4bml3uzull0ckbu/model.tgz?dl=0
	tar zxvf model.tgz
### 如果需要training的話，下載google pretrained word vectors:
	cd data
	wget -O GoogleNews-vectors-negative300.bin.gz https://www.dropbox.com/s/mgfumwi3bsh7bhe/GoogleNews-vectors-negative300.bin.gz?dl=0
	gzip -d GoogleNews-vectors-negative300.bin.gz
	cd ..

## Testing 
### Run model:
#### 測試Train的model(XXXX 輸入stored model的編號(step) 例如: 2000):
python3 inference.py --save_dir=./save/  --test_file=./data/testing_data.csv --output=pred.csv --resume_model=XXXX

#### 或是直接測試pre-train model:
python3 inference.py --save_dir=./model/ --test_file=./data/testing_data.csv --output=pred.csv

### 算accuracy:
python3 acc.py -i pred.csv

## Reference
- chiawen's [work](https://github.com/chiawen/sentence-completion)
- luser's [work](https://github.com/hunkim/word-rnn-tensorflow)
