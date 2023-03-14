from lib_dialog import logging
# Arguments


class Args():
    def __init__(self):
        self.output_dir = 'new_data_all'
        self.model_type = 'gpt2'
        self.model_name_or_path = 'byeongal/Ko-DialoGPT'
        self.config_name = 'byeongal/Ko-DialoGPT'
        self.tokenizer_name = 'byeongal/Ko-DialoGPT'

        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 2
        self.per_gpu_eval_batch_size = 2
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1  # epochs 조절
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_steps = 233060
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        # self.should_continue = True인 경우 check point 이어서 실행 (epochs가 다를 경우 이어서 학습)
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'


args = Args()
# Configs
logger = logging.getLogger(__name__)
