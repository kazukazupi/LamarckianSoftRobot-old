import glob
import os
import shutil
import argparse

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('src_exp_dir', type=str)
    parser.add_argument('dst_exp_dir', type=str)
    args = parser.parse_args()

    os.mkdir(args.dst_exp_dir)

    # copy generation00 directory
    shutil.copytree(
        src=os.path.join(args.src_exp_dir, 'generation00'),
        dst=os.path.join(args.dst_exp_dir, 'generation00')
    )

    # copy config.json
    shutil.copy(
        src=os.path.join(args.src_exp_dir, 'config.json'),
        dst=os.path.join(args.dst_exp_dir, 'config.json')
    )

    # copy fitness.csv
    shutil.copy2(
        src=os.path.join(args.src_exp_dir, 'fitness.csv'),
        dst=os.path.join(args.dst_exp_dir, 'fitness.csv')
    )

    # copy log.txt
    shutil.copy2(
        src=os.path.join(args.src_exp_dir, 'log.txt'),
        dst=os.path.join(args.dst_exp_dir, 'log.txt')
    )


if __name__ == "__main__":
    main()