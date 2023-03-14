#1: 현재 CUDA Version에 맞는 Pytorch 설치
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html pytorch_lightning==1.2.3

#2 : KoBART 모델 설치
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart

#3 : BLEU Score 설치
pip install "sacrebleu[ko]"

#4 : KoBART 실행에 필요한 파일(pretrained tokenizer... etc)
git clone --recurse-submodules https://github.com/haven-jeon/KoBART-chatbot.git

#5 : 필요한 압축파일 설치
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
get_kobart_tokenizer(".")
get_pytorch_kobart_model(cachedir=".")

#6 : wandb 설치
pip install wandb
wandb login [자신의 API Key]


# 코랩에서 돌릴려면 위 #1번이 아닌 다음 설치 명령어 실행
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html pytorch_lightning==1.2.3
!pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
!pip install datasets
!pip uninstall torchtext
!pip install wandb
!pip install "sacrebleu[ko]"
!pip install transformers==4.26.0