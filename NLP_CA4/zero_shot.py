import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


dataset = load_dataset("nyu-mll/multi_nli")


model_name = "your-llama-3-8b-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def format_prompt(premise, hypothesis):
    return f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"


def preprocess_function(examples):
    inputs = [format_prompt(premise, hypothesis)
              for premise, hypothesis in
              zip(examples['premise'], examples['hypothesis'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = examples["label"]
    return model_inputs


encoded_dataset = dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir = "./results",
    evaluation_strategy = "epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 3,
    weight_decay = 0.01,
    logging_dir = './logs',
    logging_steps = 10,
    save_total_limit = 2,
    save_steps = 1000,
    temperature = 0.7,  
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["validation_matched"],
    tokenizer = tokenizer,
)

trainer.train()

model.save_pretrained("./fine_tuned_llama_3_8b")
tokenizer.save_pretrained("./fine_tuned_llama_3_8b")

results = trainer.evaluate()
print(results)
