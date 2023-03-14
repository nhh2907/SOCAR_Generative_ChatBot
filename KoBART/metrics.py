from lib_chat_kobart import (tqdm, load_metric, np, pickle)
from setting import chatbot_args, today

# BLEU Score 함수
def blue_score(test_file, model):
    test_file.reset_index(inplace=True)             # 테스트 파일 인덱스 재정렬
    bleu_metric = load_metric("sacrebleu")          # BLEU 메트릭 객체 불러오기

    scores = []
    for i in tqdm(range(len(test_file))): 
        Q = test_file['Q'][i]
        A = test_file['A'][i]

        pred = model.chat(Q).replace('<usr> ', '')  # 모델 예측
        references = []                            
        references.append(A.strip())               # 참조 문장
        
        # BLEU Score 함수
        bleu_metric.add(prediction=pred, reference=references)
        results = bleu_metric.compute(smooth_method='exp', tokenize='ko-mecab')
        scores.append(results['score'])
        
    mean_score = np.mean(scores)
    mean_score = round(mean_score, 2)
    print('BLEU 평균 :', mean_score)

    return mean_score