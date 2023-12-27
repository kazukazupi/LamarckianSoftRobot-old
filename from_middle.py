from ga import run_from_middle
import argparse

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str)
    parser.add_argument(
        '-c', '--crossover-rate', type=float, default=None, required=False)
    parser.add_argument(
        '-i', '--inherit-en', type=bool, default=None, required=False)
    args = parser.parse_args()
    
    run_from_middle(args.exp_dir, args.crossover_rate, args.inherit_en)

if __name__ == "__main__":
    main()