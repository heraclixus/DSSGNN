r"""A logger related utils file: tqdm style, logger loader.
"""
import logging
import os
from datetime import datetime

try:
    from cilog import create_logger
except ImportError:
    def create_logger(name='GNN_log', file=None, enable_mail=False, sub_print=True, use_color=True):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers = []

        formatter = logging.Formatter('%(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if file:
            os.makedirs(os.path.dirname(file), exist_ok=True)
            file_handler = logging.FileHandler(file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        return logger
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def close(self):
            pass

pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                'dynamic_ncols': True, 'ascii': '░▒█'}

from GOOD.utils.config_reader import Union, CommonArgs, Munch


def load_logger(config: Union[CommonArgs, Munch], sub_print=True):
    r"""
    Logger loader

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.log_path`, :obj:`config.tensorboard_logdir`, :obj:`config.log_file`)
        sub_print (bool): Whether the logger substitutes general print function. If Ture, logger.info will be equal to
            print(f'#IN#Message'), where #IN# represents info. Similarly, other level of log can be used by adding prefixes
            (Not capital sensitive): Debug: #d#, #De#, #Debug#, etc. Info: #I#, #In#, #inf#, #INFO#, etc. Important: #IM#,
            #important#, etc. Warning: #W#, #war#, etc. Error: #E#, #err#, etc. Critical: #C#, #Cri#, #critical#, etc. If
            there is no prefix, the general print function will be used.

    Returns:
        [cilog Logger, tensorboard summary writer]

    """
    if sub_print:
        print("This logger will substitute general print function")
    logger = create_logger(name='GNN_log',
                           file=config.log_path,
                           enable_mail=False,
                           sub_print=sub_print)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(
        log_dir=os.path.join(config.tensorboard_logdir, f'{config.log_file}_{current_time}'))
    return logger, writer
