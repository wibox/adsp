echo "Removing previously saved models..."
rm models/trained_models/*
echo "Removing previous logs..."
rm -rf tb_logs/*

# echo "Training and testing vanilla unet"
python3 test_unet_vanilla.py

# echo "Fine tuning a Unet pretrained on BigEarthNet"
python3 test_bigearthnet.py

# echo "Fine tuning a Unet pretrained on ImageNet"
python3 test_imagenet.py

# echo "Training on EFFIS and testing on EMS"
python3 test_effis_format.py