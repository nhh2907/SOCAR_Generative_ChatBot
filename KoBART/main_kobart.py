from lib_chat_kobart import (DataLoader, Dataset, pd, logging, os, torch, pl, pl_loggers, datetime, pickle, json, 
                             WandbLogger, wandb)
from setting import chatbot_args, today
from preprocessing import *
from kobart_chit_chat import KoBARTConditionalGeneration, ChatDataModule, ChatDataset
from metrics import blue_score

if __name__ == '__main__':

    # DB에서 dataframe 넘겨 받기
    df_db = pd.read_csv(chatbot_args.train_data_for_split) # 임시로 저장되어 있는 메모데이터 불러옴

    # 전처리 실행
    ## Train, Valid data로 분리
    df_train, df_valid = preprocessing(df_db)

    # train, test 파일 저장
    df_train.to_csv(chatbot_args.train_file)
    df_valid.to_csv(chatbot_args.valid_file)  
    
    # 로깅
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(chatbot_args)

    # 모델 불러오기
    model = KoBARTConditionalGeneration(chatbot_args)
    
    # 모델 가중치 불러오기
    if os.path.isfile(chatbot_args.chatbot_model_path):
        chatbot_args.checkpoint = torch.load(chatbot_args.chatbot_model_path)
        model.load_state_dict(chatbot_args.checkpoint['model_state_dict'])    # 옵티마이저 가중치는 configure_optimizers 메소드에서 불러옴
        chatbot_args.optim_para_load = True                                   # configure_optimizers메소드에서 옵티마이저 가중치 불러오기 실행 조건
        
    # 데이터 모듈
    dm = ChatDataModule(chatbot_args.train_file,
                        chatbot_args.valid_file,
                        chatbot_args.eval_file,
                        os.path.join(chatbot_args.tokenizer_path, 'model.json'),
                        max_seq_len=chatbot_args.max_seq_len,
                        batch_size=chatbot_args.batch_size,
                        num_workers=chatbot_args.num_workers)
    
    # Pytorch Lightning 체크포인트 콜백
    ## https://pytorch-lightning.readthedocs.io/en/staable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=chatbot_args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1,
                                                       period=5,
                                                       prefix='kobart_chitchat')
    
    
    # 로거
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(chatbot_args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    wandb_logger = WandbLogger()
    
    # 학습
    trainer = pl.Trainer.from_argparse_args(chatbot_args, logger=wandb_logger, callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)                               # train, valid 실행

    #저장
    if not os.path.exists('finetuned_model'):
        os.makedirs('./finetuned_model')
    torch.save({'model_state_dict': model.state_dict(),
               'optimizer_state_dict': model.configure_optimizers()[0][0].state_dict()}, 
               chatbot_args.chatbot_model_path)
    
    
    print('>>> Evaluation Start!!!!!!')

    ## BLEU Score 성능 평가
    print('>>> BLEU Score 성능 평가 시작!!!!!!')
    df_eval = pd.read_csv(chatbot_args.eval_file)        # 추론 데이터 불러오기
    b_score = blue_score(df_eval[:2], model) 
    wandb.log({'BLEU_Score' : b_score})
    
    ## Perplexity 성능 평가
    print('>>> Perplexity 성능 평가 시작!!!!!!')   
    trainer.test(model, dm.test_dataloader())            # test 실행
    ppl_score = trainer.logged_metrics['perplexity']   
