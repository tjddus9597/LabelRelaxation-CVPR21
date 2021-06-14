cd ./pretrained_source

mkdir bn_inception
# BN-Inception (d=512) trained with Proxy-Anchor on CUB-200-2011
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuTF1AI5yHVVHT0RzBqTEnYHr5TbEdC3' -O cub_bn_inception_512dim_Proxy_Anchor_ckpt.pth
mv cub_bn_inception_512dim_Proxy_Anchor_ckpt.pth bn_inception/

# BN-Inception (d=512) trained with Proxy-Anchor on Cars-196
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ENSpNad0Zu8d6D5pMG5fsK2bU2z4KuMv' -O cars_bn_inception_512dim_Proxy_Anchor_ckpt.pth
mv cars_bn_inception_512dim_Proxy_Anchor_ckpt.pth bn_inception/

# BN-Inception (d=512) trained with Proxy-Anchor on SOP
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gyMei5IMSFshqqt0UfN5-po41JbNFULl' -O SOP_bn_inception_512dim_Proxy_Anchor_ckpt.pth
mv SOP_bn_inception_512dim_Proxy_Anchor_ckpt.pth bn_inception/

mkdir resnet50
# ResNet50 (d=512) trained with Proxy-Anchor on CUB-200-2011
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pDBfV3vNTxlYkwsMZylQEgOwmg_bicjV' -O cub_resnet50_512dim_Proxy_Anchor_ckpt.pth
mv cub_resnet50_512dim_Proxy_Anchor_ckpt.pth resnet50/

# ResNet50 (d=512) trained with Proxy-Anchor on Cars-196
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WSeBPi61NIr-dsTuZFRoUZ3RJ-S86nfS' -O cars_resnet50_512dim_Proxy_Anchor_ckpt.pth
mv cars_resnet50_512dim_Proxy_Anchor_ckpt.pth resnet50/

# ResNet50 (d=512) trained with Proxy-Anchor on SOP
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=152JIJzzFuz_v6vLtlpybB4OhphE9icRz' -O SOP_resnet50_512dim_Proxy_Anchor_ckpt.pth
mv SOP_resnet50_512dim_Proxy_Anchor_ckpt.pth resnet50/

cd .././
