`python run_benchmark.py --model EleutherAI/gpt-neo-125M --dataset rchu233/ni-ood-dataset-20250131-modernbert-train-kmeans-dim128-20250312 --context_length 512 --num_domains 60 --num_rounds 10 --training_iterations 2000 --batch_size 8 --eval_fraction 0.05`


`python run_benchmark.py --model EleutherAI/gpt-neo-125M --dataset NamburiSrinath/ni-unique-100-tasks-modernbert-train-kmeans-dim768-20250317 --context_length 512 --num_domains  38 --num_rounds 10 --training_iterations 2000 --batch_size 16 --eval_fraction 0.05 --delta 0.0005 --num_layers 73`
