
Training: 
python train.py flowers --save_dir . --arch vgg16 --learning_rate 0.001 --hidden_units 102 --gpu

--arch:  vgg16 or alexnet

Prediction:
python predict.py flowers/test/69/image_05959.jpg model_checkpoint.pth --top_k 10 --category_names cat_to_name.json --gpu