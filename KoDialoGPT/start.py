from lib_dialog import (torch, AutoTokenizer, 
                        AutoModelForCausalLM, transformers)

#ignore warning messages
transformers.logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained('new_data_all')
model = AutoModelForCausalLM.from_pretrained('new_data_all')

while True:
    step = 0
    user_str=input(">> 사용자:")
    new_user_input_ids = tokenizer.encode(user_str + tokenizer.eos_token, return_tensors='pt')
    if user_str=='끝':
        break
    # append the new user input tokens to the chat histocry
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000,
            pad_token_id=tokenizer.eos_token_id,  
            no_repeat_ngram_size=100,       
            do_sample=True, 
            top_k=200, 
            top_p=0.3,
            temperature=0.8
        )
        # pretty print last ouput tokens from bot
        print("쏘카봇: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))