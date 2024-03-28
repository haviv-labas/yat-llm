from app.utils import CfgNode as CN
from skopt.space import Real, Integer


def get_model_config():
    C = CN()
    C.model_type = "feedforward"
    C.data_type = "chars"
    C.n_embd = None
    C.vocab_size = None
    C.block_size = None

    return C


def get_trainer_config():
    C = CN()
    C.device = "auto"

    # dataloder parameters
    C.num_workers = 4

    # optimizer parameters
    C.max_iters = 1000
    C.batch_size = 64
    C.learning_rate = 5e-4
    C.betas = (0.9, 0.95)
    C.weight_decay = 0.1
    C.grad_norm_clip = 1.0
    C.split_ratio = 0.7
    C.maximum_validation_block_size = 200

    return C


def get_all_config():
    C = CN()
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/chargpt"

    # model
    C.model = get_model_config()

    # trainer
    C.trainer = get_trainer_config()

    return C


def optimisation_space():
    return [
        Real(1e-5, 1e-2, name="learning_rate"),
        Integer(48, 128, name="n_embds"),
        Integer(500, 5000, name="epochs"),
        Integer(10, 100, name="block_size"),
        Integer(1, 5, name="depth"),
        Integer(6, 12, name="width"),
    ]
