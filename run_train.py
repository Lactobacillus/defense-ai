import os
import sys
import codecs
import argparse
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

from train.defense import Trainer

def main(args: Dict[str, Any]) -> None:

    match args['stage']:

        case 1:

            # Pretraine VideoMAE
            from train.defense import PreTrainer

            trainer = Trainer(args)
            trainer.train('train')

        case 2:

            # Finetune VideoMAE
            from train.defense import FinetuneTrainer
            
            trainer = Trainer(args)
            trainer.train('train')

        case 3:

            # Finetue VideoMAE using KD and reset
            from train.defense import FinetuneTrainerKD
            
            trainer = Trainer(args)
            trainer.train('train')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type = str,
                        help = 'config file')
    parser.add_argument('--stage', type = int, choices = [1, 2, 3],
                        help = 'select training stage')
    parser.add_argument('--omp-num-threads', type = int,
                        default = 2,
                        help = 'OMP_NUM_THREADS option')
    opt = vars(parser.parse_args())

    with codecs.open(opt['conf'], 'r', encoding = 'UTF-8') as fs: exec(fs.read())

    os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args)
