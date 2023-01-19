echo "Training and testing vanilla unet"
python3 test_unet_vanilla.py

echo "Fine tuning a Unet pretrained on BigEarthNet"
python3 test_bigearthnet.py

echo "Fine tuning a Unet pretrained on ImageNet"
python3 test_imagenet.py

echo "Computing output for three different models..."
python3 compute_test_output.py