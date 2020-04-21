#!/bin/sh

case "$1" in
"LSTMTransformer")
  echo "Train LSTM+Transformer model."
  python -O src/train.py \
    --data data/wikitext-103 \
    --model LSTMTransformer \
    --emsize 512 \
    --nhid 1024 \
    --nlayers 5 \
    --clip 5.0 \
    --epochs 5 \
    --batch_size 16 \
    --bptt 64 \
    --dropout 0.1 \
    --seed 1111 \
    --cuda \
    --val-interval 5000 \
    --log-interval 100 \
    --save models/lstmtransformer.pth \
    --tb-log logs/lstmtransformer \
    --nhead 8
  ;;
"Transformer")
  echo "Train Transformer model."
  python -O src/train.py \
    --data data/wikitext-103 \
    --model Transformer \
    --emsize 512 \
    --nhid 1024 \
    --nlayers 6 \
    --clip 5.0 \
    --epochs 5 \
    --batch_size 16 \
    --bptt 64 \
    --dropout 0.1 \
    --seed 1111 \
    --cuda \
    --val-interval 5000 \
    --log-interval 100 \
    --save models/transformer.pth \
    --tb-log logs/transformer \
    --nhead 8
  ;;
"LSTM")
  echo "Train LSTM model."
  python -O src/train.py \
    --data data/wikitext-103 \
    --model LSTM \
    --emsize 512 \
    --nhid 512 \
    --nlayers 3 \
    --step-size 1 \
    --clip 5.0 \
    --epochs 5 \
    --batch_size 16 \
    --bptt 64 \
    --dropout 0.2 \
    --seed 1111 \
    --cuda \
    --val-interval 5000 \
    --log-interval 100 \
    --save models/lstm.pth \
    --tb-log logs/lstm
  ;;
*)
  echo "Usage:"
  echo "./train.sh {LSTMTransformer,Transformer,LSTM}"
  ;;
esac
