from basicsr.utils.logger import MessageLogger


class FakeTensorBoardLogger:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def make_logger(tb_logger):
    opt = {
        'name': 'BioIR-LOLv1',
        'logger': {'print_freq': 100, 'use_tb_logger': True},
        'train': {'total_iter': 300000},
    }
    return MessageLogger(opt, start_iter=0, tb_logger=tb_logger)


def test_tensorboard_loss_uses_current_iter_as_step():
    tb_logger = FakeTensorBoardLogger()
    logger = make_logger(tb_logger)

    logger({
        'epoch': 8,
        'iter': 500,
        'total_iter': 300000,
        'lrs': [1e-3],
        'time': 0.42,
        'data_time': 0.01,
        'l_pix': 0.12,
    })

    assert ('losses/l_pix', 0.12, 500) in tb_logger.scalars


def test_tensorboard_records_learning_rate_and_timing():
    tb_logger = FakeTensorBoardLogger()
    logger = make_logger(tb_logger)

    logger({
        'epoch': 8,
        'iter': 500,
        'total_iter': 300000,
        'lrs': [1e-3],
        'time': 0.42,
        'data_time': 0.01,
        'l_total': 0.18,
    })

    assert ('train/lr_g_0', 1e-3, 500) in tb_logger.scalars
    assert ('time/iter', 0.42, 500) in tb_logger.scalars
    assert ('time/data', 0.01, 500) in tb_logger.scalars
