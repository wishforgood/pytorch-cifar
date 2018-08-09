import os

# lr = 0.25
# resume = ''
# os.system(
#     'python /root/mounted_device/PycharmProjects/Inclusive_Classification/fghp-CIFAR-10.py --lr ' + str(lr) + ' ' + str(
#         resume))

import multiprocessing

resume = '--resume'


def my_execute(lr):
    os.system(
        'python /root/mounted_device/PycharmProjects/Inclusive_Classification/fghp-CIFAR-10.py --lr ' + str(
            lr) + ' ' + str(
            resume))


p = multiprocessing.Pool()
p.map(my_execute, [0.06, 0.05])
