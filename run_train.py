import os
import sys
import codecs
import argparse
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable


def main(args: Dict[str, Any],
        stage: int) -> None:

    match stage:

        case 1:

            from train.new_defense import Stage1Trainer as Trainer

            trainer = Trainer(args)
            trainer.train('train')

        case 2:

            from train.defense2 import Stage2Trainer as Trainer
            
            trainer = Trainer(args)
            trainer.train('train')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type = str,
                        help = 'config file')
    parser.add_argument('--stage', type = int, choices = [1, 2],
                        help = 'training stage: [1, 2, 3]')
    parser.add_argument('--omp-num-threads', type = int,
                        default = 2,
                        help = 'OMP_NUM_THREADS option')
    opt = vars(parser.parse_args())

    with codecs.open(opt['conf'], 'r', encoding = 'UTF-8') as fs: exec(fs.read())

    os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args, stage = opt['stage'])
