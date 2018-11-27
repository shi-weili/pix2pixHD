#!/bin/bash
################################ Testing ################################
# labels only
python test.py --name label2city_1024p --netG local --ngf 32 --loadSize 4096 --fineSize 4096 $@
