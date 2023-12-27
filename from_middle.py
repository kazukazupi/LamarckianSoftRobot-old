from ga import run_from_middle
import argparse

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str)
    args = parser.parse_args()
    
    run_from_middle(args.exp_dir)

if __name__ == "__main__":
    main()