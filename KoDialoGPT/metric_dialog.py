
from lib_dialog import (
    AutoTokenizer, AutoModelForCausalLM,
    load_metric, pd, tqdm, torch, np, transformers
)
# bleu 평가


def metric_bleu():
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained('byeongal/Ko-DialoGPT')
    model = AutoModelForCausalLM.from_pretrained('new_data_all')

    test_file = pd.read_csv('new_test.csv')  # 경로 설정
    bleu_metric = load_metric("sacrebleu")
    scores = []
    for i in tqdm(range(len(test_file))):
        step = 0
        q = test_file['Q'][i]
        a = test_file['A'][i]

        new_user_input_ids = tokenizer.encode(
            q + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat(
            [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        # 예측한 문장
        pred = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        references = []
        references.append(a)             # 참조 문장

        bleu_metric.add(prediction=pred, reference=references)
        results = bleu_metric.compute(smooth_method='exp', tokenize='ko-mecab')
        results['precisions'] = [np.round(p, 2) for p in results['precisions']]
        scores.append(results['precisions'])

    return (np.mean(scores))
