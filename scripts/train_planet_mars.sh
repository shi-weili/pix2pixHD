# python3 train.py --planet_mars --name planet_mars_gen_2 --loadSize 128 --fineSize 128 --label_nc 0 --input_nc 7 --dataroot 'datasets/seamless_grid' --no_flip --no_instance --batchSize 8

# python3 train.py --planet_mars --name planet_mars_gen_2 --loadSize 512 --fineSize 512 --label_nc 0 --input_nc 7 --dataroot 'datasets/seamless_grid' --no_flip --no_instance --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7

python3 train.py --planet_mars --name planet_mars_gen_2 --loadSize 512 --fineSize 512 --label_nc 0 --input_nc 7 --dataroot 'datasets/seamless_grid' --no_flip --no_instance

sudo shutdown -h now