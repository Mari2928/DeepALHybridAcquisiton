The codes are adapted from https://github.com/lunayht/DBALwithImgData. There are annotations in the codes to help understand the logic.

The diversity metric names in the codes are different from those in the report. Here is the mapping:

discriminator score - waal

posterior variance - density

minimum distance - minidis

To run the constant weighted arithmetic mean:

$ python main.py --uncertainty 100 --diversity 100 --sum_product 1

To run the time-decayed weighted arithmetic mean:

$ python main.py --uncertainty 100 --diversity 100 --sum_product 1 --time_decay True

To run the constant weighted geometric mean:

$ python main.py --uncertainty 100 --diversity 100 

To run the time-decayed weighted geometric mean:

$ python main.py --uncertainty 100 --diversity 100 --time_decay True

To run the uncertainty-first two stage query:
    
$ python main.py --uncertainty 100 --diversity 100 --runmode 1

To run the diversity-first two stage query:
    
$ python main.py --uncertainty 100 --diversity 100 --runmode 1 --priority 1

To get baseline performances:

#entropy 

$ python main.py --beta 1

#bald

$ python main.py --beta 1 --uncertainty 1

#variation ratio

$ python main.py --beta 1 --uncertainty 2

#discriminator score

$ python main.py --beta 0

#posterior variance

$ python main.py --beta 0 --diversity 1

#minimum distance

$ python main.py --beta 0 --diversity 2

#uniform

$ python main.py --uncertainty 10 --diversity 10


