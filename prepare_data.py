import gzip
import pickle as pkl
import argparse
from tqdm import tqdm
from random import randint
from collections import Counter
import sys


def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def build_dataset(infile,rescale=False,nb_splits=5):

	def preprocess(rev_dict):

		rating = rev_dict["overall"]
		if rescale:
			if rating > 3:
				rating = 1
			elif rating == 3:
				return None
			else:
				rating = 0

		return (rev_dict['reviewText'],rating)


	print("Building dataset from : {}".format(infile))
	print("-> Building {} random splits".format(nb_splits))
	if rescale:
		print("-> Rescaling data to 0-1 (3's are discarded)")


	with gzip.open(infile,"r") as f:
		data = [eval(x) for x in tqdm(f,desc="getting reviews",total=count_lines(f))]
		data = [preprocess(rev) for rev in tqdm(data,desc="preprocessing")]
		data = [d for d in data if d is not None]

		splits = [randint(0,nb_splits-1) for _ in range(0,len(data))]
		count = Counter(splits)

		print("Split distribution is the following:")
		print(count)

		return {"data":data,"splits":splits,"rows":("review","rating")}



def main(args):
    ds = build_dataset(args.input,args.rescale,args.nb_splits)
    pkl.dump(ds,open(args.output,"wb"))
    
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", type=str, default="sentences.pkl")
parser.add_argument("--rescale",action="store_true")
parser.add_argument("--nb_splits",type=int, default=5)


args = parser.parse_args()

main(args)