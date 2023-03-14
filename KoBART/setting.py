from lib_chat_kobart import yaml, argparse, timezone, datetime

# 맞춤법 검사기
# check_spell_url = "http://164.125.7.61/speller/results"

# 한국
KST = timezone('Asia/Seoul')
today = datetime.now().astimezone(KST)

# Hyperparameters
chatbot_hparams = 'hparams.yaml'
with open(chatbot_hparams) as f:
    chatbot_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
    chatbot_args.optim_para_load = False
    
    chatbot_args.train_data_for_split = 'data/qa_faq_train_ver1.csv' # train, valid으로 분리될 데이터 
    chatbot_args.train_file = 'data/train.csv'                       # 분리된 train 데이터
    chatbot_args.valid_file = 'data/valid.csv'                       # 분리된 valid 데이터
    chatbot_args.eval_file = 'data/qa_faq_test_ver1.csv'             # 모델 eval 데이터
    
    chatbot_args.chatbot_model_path = 'finetuned_model/checkpoint_chatbot_model_weights.pth'  # 모델, 옵티마이저 가중
    chatbot_args.finetuned_dir = './finetuned_model'                                          # 경로 생성
        
    chatbot_args.batch_size = 32
    chatbot_args.lr = 5.0e-05
    chatbot_args.max_epochs = 1
    chatbot_args.max_seq_len = 128
    chatbot_args.chat = True