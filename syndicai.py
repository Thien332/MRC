import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM



class PythonPredictor:
    def __init__(self, config):
        """This method is required. It is called once before the API 
        becomes available. It performes the setup such as downloading / 
        initializing the model.

        :param config (required): Dictionary passed from API configuration.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")

        self.device = device
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base',return_dict = True)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')




    def predict(self, payload):
        """This method is required. It is called once per request. 
        Preprocesses the request payload, runs inference, and 
        postprocesses the inference output.

        :param payload (optional): The request payload
        :returns : Prediction or a batch of predictions.
        """
        # gen_tokens = self.model.generate(payload["text"], do_sample=True, temperature=0.9, max_length=100)
        # self.tokenizer.batch_decode(gen_tokens)[0]
        inputs = self.tokenizer.encode_plus(payload["text"], return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)

        token_logits = self.model(**inputs).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
        # f_token = self.tokenizer.decode([top_tokens])[0]
        for token in top_tokens:
            word = self.tokenizer.decode([token])
            new_sentence = payload["text"].replace(self.tokenizer.mask_token, word)
            # print(new_sentence)
        

        return new_sentence
