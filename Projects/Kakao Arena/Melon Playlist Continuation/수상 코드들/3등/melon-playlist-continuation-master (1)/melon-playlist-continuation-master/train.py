import argparse

import w2v
from w2v import LossPrinter
import song_title_to_tag

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default="all")
  args = parser.parse_args()

  if args.mode not in ['all', 'tag', 'vec']:
    print("Invalid argument for --mode, input 'all', 'tag' or 'vec'")

  if args.mode in ['all', 'tag']:
    print("Extracting tags from song titles")
    song_title_to_tag.extract_tag_from_song_title()
  if args.mode in ['all', 'vec']:
    print("Training Word2Vec")
    w2v.train()
  print("Ends")