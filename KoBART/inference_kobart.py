from lib_chat_kobart import (yaml, torch, os)
from setting import chatbot_args
from kobart_chit_chat import KoBARTConditionalGeneration

# 모델 불러오기
model = KoBARTConditionalGeneration(chatbot_args)

# 모델 가중치 불러오기
if os.path.isfile(chatbot_args.chatbot_model_path):
    chatbot_args.checkpoint = torch.load(chatbot_args.chatbot_model_path)
    model.load_state_dict(chatbot_args.checkpoint['model_state_dict'])     # 옵티마이저 가중치는 configure_optimizers 메소드에서 불러옴


# 챗봇 추론
if chatbot_args.chat:
    model.model.eval()
    while 1:
        q = input('user > ').strip()
        if q == 'quit':
            break
        print("쏘카봇 > {}".format(model.chat(q).replace('<usr> ','')))