import datasets
import random
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers.integrations import WandbCallback
from transformers import GenerationConfig
from tqdm import tqdm
import wandb
import trl


class LLMSampleCB(WandbCallback):
  def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
      "A CallBack to log samples a wandb.Table during training"
      super().__init__()
      self._log_model = log_model
      self.sample_dataset = test_dataset.select(range(num_samples))
      self.model, self.tokenizer = trainer.model, trainer.tokenizer
      self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                          max_new_tokens=max_new_tokens,
                                                          temperature=0.7,
                                                          repetition_penalty=1.2)
  def generate(self, prompt):
      tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
      with torch.inference_mode():
          output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
      return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):])

  def samples_table(self, examples):
      "Create a wandb.Table to store the generations"
      records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
      for example in tqdm(examples, leave=False):
          prompt = example["text"]
          generation = self.generate(prompt=prompt)
          records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
      return records_table

  def on_evaluate(self, args, state, control,  **kwargs):
      "Log the wandb.Table after calling trainer.evaluate"
      super().on_evaluate(args, state, control, **kwargs)
      records_table = self.samples_table(self.sample_dataset)
      self._wandb.log({"sample_predictions_step_" +str(state.global_step):records_table})


project_name = "mixtral-claire-lora-4bits-r8"

def reduce_test(exemple):
  split = random.randint(200, 300)
  exemple['text'] = exemple['orig'][0:split]
  return exemple


dataset = datasets.load_dataset('OpenLLM-France/Claire-Dialogue-French-0.1')
dataset['test'] = dataset['test'].add_column("orig", dataset['test']['text'])
dataset['test'] = dataset['test'].map(reduce_test)
dataset['test'] = dataset['test'].remove_columns(['orig'])


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",
                                             load_in_4bit=True,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                            )

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

tokenizer.pad_token = "!" #Not EOS, will explain another time.\

CUTOFF_LEN = 1024  #Our dataset has shot text
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[ "w1", "w2", "w3"],  #just targetting the MoE layers.
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# dataset = load_dataset("PhilSad/Claire-Dialogue-French-0.1")
print("dataset", dataset)
train_data = dataset["train"]
test_data  = dataset["test"]

train_data = train_data.shuffle()
test_data = test_data.shuffle()




trainer = trl.SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=config,
    dataset_text_field="text",
    max_seq_length=CUTOFF_LEN,
    tokenizer=tokenizer,
    args=TrainingArguments(
        report_to="wandb",
        bf16=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        optim="adamw_torch",
        save_strategy="steps",
        output_dir=project_name,
        push_to_hub=True,
        save_total_limit=5,
        logging_strategy="steps",
        evaluation_strategy="steps",
        eval_steps = 100,
        logging_steps=1,
        save_steps = 100

    )
)
model.config.use_cache = False


wandb.init(project=project_name, config = dict(peft_config=config.to_dict()))
wandb_callback = LLMSampleCB(trainer, test_data, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)
trainer.train()

wandb.finish()
