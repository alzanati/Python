'''
@author: Alzanati
@Date: 22 / 4 / 2016
'''

import subprocess

f = raw_input("Enter sha256sum of ubuntu 16.04 : ")
path = raw_input("path of ubuntu 16.04 iso : ")

p = subprocess.Popen(["sha256sum", path], stdout=subprocess.PIPE)
(output, err) = p.communicate()
output = output[:-len(path)-3]
print (output)

if output == f:
	print ('\n!!!!!!!!!!!!!!! True Matching !!!!!!!!!!!!!!!!!\n')
else:
	print ('\n!!!!!!!!!!!!!!!!! False Matching !!!!!!!!!!!!!!!!!\n')
