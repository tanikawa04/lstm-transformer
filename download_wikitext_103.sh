#!/bin/sh

cd data

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
mv wikitext-103/wiki.train.tokens wikitext-103/train.txt
mv wikitext-103/wiki.valid.tokens wikitext-103/valid.txt
mv wikitext-103/wiki.test.tokens wikitext-103/test.txt

cd ..

echo "Done."
