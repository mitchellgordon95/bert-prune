from tensorflow.python.training import checkpoint_management
import re
import glob
import fire
import os

def delete_all_but_last(*model_dirs):
    """Deletes all checkpoints except for the last one"""
    for model_dir in model_dirs:
        latest = checkpoint_management.latest_checkpoint(model_dir)
        try:
            prefix = re.match(r'(.*)-\d*$', latest).group(1)
        except:
            continue
        for path in glob.glob(prefix+'*'):
            if not path.startswith(latest):
                print(f'Removing {path}')
                try:
                    os.remove(path)
                except:
                    print(f'Unable to remvoe {path}')


if __name__ == '__main__':
    fire.Fire(delete_all_but_last)
