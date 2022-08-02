from transformers import BertTokenizer, BertForPreTraining, BertForQuestionAnswering, BertModel, BertConfig
from transformers import XLMRobertaForQuestionAnswering, XLMRobertaTokenizer
import torch
import torch.nn as nn
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, SquadProcessor, SquadExample
from transformers.data.processors.squad import squad_convert_examples_to_features
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange, tqdm



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
        self.model = RobertaForQuestionAnswering.from_pretrained("./final_model").to(device)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.processor= SquadV1Processor()
    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    def evaluate(model, tokenizer, dev_dataset, dev_examples, dev_features):
        eval_sampler = SequentialSampler(dev_dataset)
        eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=32)
        all_results = []
    #     start_time = timeit.default_timer()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
    #                "token_type_ids": batch[2],
                }
                example_indices = batch[3]
                outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = dev_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs]
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )
                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

        output_prediction_file = os.path.join("./", "predictions_{}.json".format(""))
        output_nbest_file = os.path.join("./", "nbest_predictions_{}.json".format(""))
        output_null_log_odds_file = os.path.join("./", "null_odds_{}.json".format(""))
        predictions = compute_predictions_logits(
                dev_examples,
                dev_features,
                all_results,
                20,
                128,
                False,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                True,
                False,
                0.0,
                tokenizer,
            )
        return predictions

#     def get_qa(topic, data):
#         q = []
#         a = []
#         for d in data['data']:
#             if d['title']==topic:
#                 for paragraph in d['paragraphs']:
#                     for qa in paragraph['qas']:
#                         if not qa['is_impossible']:
#                             q.append(qa['question'])
#                             a.append(qa['answers'][0]['text'])
#                 return q,a

#     questions, answers = get_qa(topic='Premier_League', data=data)

    print("Number of available questions: {}".format(len(questions)))

    def predict(self, payload):
        test_examples = self.processor.get_dev_examples('',payload["text])
        test_features, test_dataset = squad_convert_examples_to_features(test_examples, 
                                                               tokenizer, 
                                                               max_seq_length = 256, 
                                                               doc_stride = 81,
                                                               max_query_length = 81,
                                                               is_training = False,
                                                               return_dataset = 'pt',
                                                               threads = 10
                                                               )
        results = evaluate(self.model, self.tokenizer, test_dataset, test_examples, test_features)
        return results
        
