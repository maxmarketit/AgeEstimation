## DEFAULT
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --learning_rate 0.002 --epoch 1000 --loss_type 1 --result_directory result-l1-e1000
## loss_type 2
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --learning_rate 0.002 --epoch 1000 --result_directory result-l2-lr-001-e3000
## Auto result-directory
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --learning_rate 0.002 --epoch 3000 #--result_directory result-l2-lr-001-e3000

## Error in LAMBDA_1 and LAMBDA_2 found : 2020-05-09

## 2020-05-09 - 2020-05-11
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.05 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.1 --lambda2 0.05 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.4 --lambda2 0.05zz --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.1 --learning_rate 0.002 --epoch 3000


## 2020-05-11 -
# Make sure --loss_type 2 works okay	
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.4 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.4 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.6 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.27 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.4 --lambda2 0.0375 --learning_rate 0.002 --epoch 1000
	
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.4 --lambda2 0.0167 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.6 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.27 --lambda2 0.05 --learning_rate 0.002 --epoch 1000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.4 --lambda2 0.0375 --learning_rate 0.002 --epoch 1000
	
#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.4 --lambda2 0.0167 --learning_rate 0.002 --epoch 1000

## 2020-05-12

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.0375 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.0167 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.27 --lambda2 0.025 --learning_rate 0.002 --epoch 3000

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.14 --lambda2 0.025 --learning_rate 0.002 --epoch 3000

## 2020-05-13
# Added argument cuda

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.05 --learning_rate 0.001 --epoch 1000 --cuda 1

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.05 --learning_rate 0.001 --epoch 10000 --cuda 1

#python main.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.037 --learning_rate 0.002 --epoch 10000 --cuda 1


## 2020-05-14
#result-l2-lr=000200-l1=0200-l2=0025-e003000
#python main_back_v03.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.002 --epoch 10000 --cuda 1

#python main_back_v03.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.001 --epoch 10000 --cuda 1

## 2020-05-15
#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.004 --epoch 10000 --cuda 1

#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.008 --epoch 10000 --cuda 1

## 2020-05-16
#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.008 --epoch 10000 --cuda 1

#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.016 --epoch 10000 --cuda 1

#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.016 --epoch 10000 --cuda 1

## 2020-05-17
#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.4 --lambda2 0.05 --learning_rate 0.016 --epoch 5000 --cuda 1

## 2020-05-18
#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 5000 --cuda 1

#python main_back_v04.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 5000 --cuda 1

## 2020-05-19
#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 2000 --cuda 1 --CNN resnet34

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 2000 --cuda 1 --CNN resnet34

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 2000 --cuda 1 --CNN resnet34

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 2000 --cuda 1 --CNN resnet34


#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152



#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152



#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTL.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152



#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt


#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152  ## WE GET NAN!!! for loss


## 2020-05-23

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet152

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet152  ## WE GET NAN!!! for loss


#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet34

#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet34  ## WE GET NAN!!! for loss


#python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 0.2 --lambda2 0.025 --learning_rate 0.032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt


## 2020-05-24

python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt

python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet152 --meta ~/dataset/FGNET/meta.txt ## WE GET NAN!!! for loss


python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 1 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet34 --meta ~/dataset/FGNET/meta.txt

python mainMTLgender.py --batch_size 64 --image_directory ~/dataset/FGNET/images --leave_subject 1 --loss_type 2 --lambda1 1 --lambda2 1 --learning_rate 0.0032 --epoch 500 --cuda 1 --CNN resnet34 --meta ~/dataset/FGNET/meta.txt ## WE GET NAN!!! for loss


telegram 'python mainMTLgender.py on cuda:0 finished'


telegram 'python mainMTLgender.py finished'







