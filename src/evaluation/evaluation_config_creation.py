from argparse import ArgumentParser
import os
import yaml

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_template', type=str)
    parser.add_argument('--model_paths', nargs="+")
    parser.add_argument('--out_config_dir', type=str)
    
    args = parser.parse_args()
    paths = args.model_paths
    out_config_dir = args.out_config_dir
    config_path = args.config_template
    with open(config_path) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    
    for i, path in enumerate(paths):
        config['data']['outpath'] = path
        with open(os.path.join(out_config_dir, f'config_{i}.yaml'), 'w') as writer:
            yaml.dump(config, writer)
        