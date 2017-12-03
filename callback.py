import time

def X():

	print("hello")


def K():

	print("goodbye")

def Y(callback): 

	time.sleep(2)
	callback()







Y(X)

Y(K)