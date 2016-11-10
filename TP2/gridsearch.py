import sys, os, subprocess
import numpy as np

def name_file(basename, a, e, g):
	return basename+'_'+str(int(a*100))+'_'+str(int(e*100))+'_'+str(int(g*100))

def main(**kwargs):
	try:
		basename = sys.argv[1]
		iterations = sys.argv[2]
		print basename
	except:
		print "Te falto algo amiwo"
		return
	for a in np.arange(0.1, 1.0, 0.1):
		for e in np.arange(0.1, 0.4, 0.1):
 			for g in np.arange(0.4, 1.0, 0.1):
 				subprocess.call("python tp2.py rows=6 cols=7 X=4 iter="+str(iterations)+" alpha="+str(a)+" epsilon="+str(e)+" gamma="+str(g)+" mode=2 filename="+name_file(basename, a, e, g), shell=True)


if __name__ == '__main__':
    main()