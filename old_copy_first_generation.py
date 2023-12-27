import argparse
import csv
import glob
import shutil
import json
import os

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('src_exp_dir', type=str)
    parser.add_argument('dst_exp_dir', type=str)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.dst_exp_dir, 'generation00')):
        os.makedirs(os.path.join(args.dst_exp_dir, 'generation00'))

    with open(os.path.join(args.src_exp_dir, 'fitness.csv')) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                first_row = row
            if i == 1:
                second_row = row
                fitness_list = [float(fitness) for fitness in row[1:]]

    with open(os.path.join(args.dst_exp_dir, 'fitness.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(first_row)
        writer.writerow(second_row)

    gen0_id_list = sorted(glob.glob(os.path.join(args.src_exp_dir, f'generation00/id*')))

    for id, src_id_dir in enumerate(gen0_id_list):

        dst_id_dir = src_id_dir.replace(args.src_exp_dir, args.dst_exp_dir)

        os.mkdir(dst_id_dir)

        shutil.copy(
            src=os.path.join(src_id_dir, 'robot_dir/structure.npy'),
            dst=os.path.join(dst_id_dir, 'body.npy'))
        shutil.copy(
            src=os.path.join(src_id_dir, 'robot_dir/connections.npy'),
            dst=os.path.join(dst_id_dir, 'connections.npy'))
        shutil.copy(
            src=os.path.join(src_id_dir, 'robot_dir/parameter.pt'),
            dst=os.path.join(dst_id_dir, 'actor_critic.pt'))
        shutil.copy(
            src=os.path.join(src_id_dir, 'log.csv'),
            dst=os.path.join(dst_id_dir, 'log.csv'))
        
        robot_info = {
            "id": id,
            "learning_en": False,
            "parents_id": [],
            "fitness": fitness_list[id]
        }

        with open(os.path.join(dst_id_dir, 'robot_info.json'), 'w') as fp:
            json.dump(robot_info, fp)

if __name__ == "__main__":
    main()
