import math
def check(n):
	if n == 1:
		return False
	for x in range(2, (int)(math.sqrt(n))+1):
		if n % x == 0:
			return False
	return True
n = 23

    """This is my example docstring

        A simple bullet list:

        - point 1
        - point 2
        - point 3

    """

if check(n):
	print("prime")
else:
	print("not prime")