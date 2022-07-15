# Train spuriously correlated models
python3 train.py --dataset="bear-bird-cat-dog-elephant:dog(snow)"
python3 train.py --dataset="bear-bird-cat-dog-elephant:dog(water)"
python3 train.py --dataset="bear-bird-cat-dog-elephant:dog(bed)"
python3 train.py --dataset="bear-bird-cat-dog-elephant:cat(keyboard)"
python3 train.py --dataset="bear-bird-cat-dog-elephant:bird(water)" 


## Evaluate mistakes of models
python3 evaluate_cce.py --dataset="bear-bird-cat-dog-elephant:dog(snow)" --concept-bank="../examples/resnet18_bank.pkl" --model-path="../examples/models/dog(snow).pth"
python3 evaluate_cce.py --dataset="bear-bird-cat-dog-elephant:dog(water)" --concept-bank="../examples/resnet18_bank.pkl" --model-path="../examples/models/dog(water).pth"
python3 evaluate_cce.py --dataset="bear-bird-cat-dog-elephant:dog(bed)" --concept-bank="../examples/resnet18_bank.pkl" --model-path="../examples/models/dog(bed).pth"
python3 evaluate_cce.py --dataset="bear-bird-cat-dog-elephant:cat(keyboard)" --concept-bank="../examples/resnet18_bank.pkl" --model-path="../examples/models/cat(keyboard).pth"
python3 evaluate_cce.py --dataset="bear-bird-cat-dog-elephant:bird(water)" --concept-bank="../examples/resnet18_bank.pkl" --model-path="../examples/models/bird(water).pth"
