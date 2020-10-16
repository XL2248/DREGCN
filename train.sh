export CUDA_VISIBLE_DEVICES=2
inter=2
gcn_layer=2
r_dim=100
cls=4
for idx in `seq 0 1 5`
do
    python train.py --domain res -b 50 --g 200 --c 128 --aspect-layers 1 --interactions $inter --shared-layers $gcn_layer --dropout 0.5 --senti-layers 1 --lr 0.0005 --use-cnn 1 --use-doc 0 --use-meanpool 0 --use-domain-emb 1 -e 400 --use-opinion 0 --relation_dim $r_dim --use-bert 0 --use-bert-cls $cls
done

