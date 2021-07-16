import argparse

import cf1, cf2, ensemble

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--testset', type=str, default="test")
  args = parser.parse_args()

  if args.testset == "test":
    MODE = "Test"
  elif args.testset == "dev":
    MODE = "Dev"
  else:
    print("Invalid argument for --testset, input 'test' or 'dev'")

  if args.testset in ["test", "dev"]:
    print("CF Model 1 infrence starts")
    cf1.infer(MODE)
    print("CF Model 2 infrence starts")
    cf2.infer(MODE)
    print("Ensemble")
    ensemble.ensemble(MODE)
  print("Ends")